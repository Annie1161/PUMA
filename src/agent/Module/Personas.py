"""
FEP-style rolling belief about the client (under the two-step mode this is
**just three long-term text state blocks** + a structured history):

1) **`_q_st_text` (short-term q)**
   Each turn, the short-term MACHINE_JSON `q_delta` performs add / modify / drop
   on the previous text via `apply_q_roll_delta`, and the result is truncated
   before being written back.

2) **`_observation_model_block` (long-term p(o|s))**
   Each turn, the long-term `observation_model_delta` performs add / modify / drop
   via `apply_long_term_block_delta`, and the result is truncated.

3) **`_transition_model_block` (long-term p(s'|s,a))**
   Each turn, the long-term `transition_model_delta` is applied the same way.

In addition: `_latent_mi`, `_observations` / `_transitions` still append items
from the payload; `_q_turn_state_probs` is updated at the end of every
`_apply_payload` to the 3-state distribution synthesized from **this turn's**
short-term inference -- so before the **next** turn's short-term LLM call, the
value is the **previous turn's** short-term estimate (uniform prior on the first
turn).

The single-step JSON mode (`--persona_single_step`) keeps the legacy one-shot
approximation; the two-step mode is the default.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from tools.persona_machine_json import (
    apply_two_step_machine_json_to_personas,
    clamp_conf,
    estimate_q_turn_state_probs_from_payload,
    parse_long_term_llm_output,
    parse_short_term_llm_output,
    render_observation_rules,
    render_q_abstract_rows,
    render_transition_rules,
)
from Module.prompts import (
    PERSONA_LONG_TERM_P_SYS,
    PERSONA_SHORT_TERM_Q_SYS,
    build_persona_extraction_sys,
    format_persona_extraction_user,
    format_persona_long_term_p_user,
    format_persona_short_term_q_user,
)

# Matches the AnnoMI corpus: 3-class client_talk_type for client utterances.
PATIENT_CLIENT_TALK_TYPES = ("neutral", "change", "sustain")

# CAMI readiness-stage 3-tuple.
CAMI_STAGE_TYPES = ("precontemplation", "contemplation", "preparation")

# Latent state keeps only this scalar (aligned with the data annotation; replaces the multi-dim MI table).
MI_LATENT_KEYS = ("client_talk_type",)

# Therapist action -- coarse granularity: `main_therapist_behaviour` (the mode column).
THERAPIST_ACTIONS_COARSE = ("question", "reflection", "therapist_input", "other")

# Therapist action -- fine granularity: `therapist_action_fine_qualified` (values seen in the corpus + "other").
THERAPIST_ACTIONS_QUALIFIED = (
    "question/open",
    "question/closed",
    "reflection/simple",
    "reflection/complex",
    "therapist_input/information",
    "therapist_input/advice",
    "therapist_input/negotiation",
    "therapist_input/options",
    "other",
)

# Therapist action -- MISC 17-class (Miller et al., 2002).
# Aligned with the MISC_LABELS used by SimpleSimulator / misc_annotator.
THERAPIST_ACTIONS_MISC17 = (
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


def therapist_action_vocab(granularity: str) -> Tuple[str, ...]:
    g = (granularity or "coarse").strip().lower()
    if g == "fine":
        return THERAPIST_ACTIONS_QUALIFIED
    if g in ("mi12", "misc17"):
        return THERAPIST_ACTIONS_MISC17
    return THERAPIST_ACTIONS_COARSE


# Backward-compatible import: defaults to the coarse vocabulary.
THERAPIST_ACTIONS = THERAPIST_ACTIONS_COARSE


@dataclass
class ObservationEntry:
    state_pattern: str = ""
    textual_cues: str = ""
    behavioral_cues: str = ""
    prediction_error: str = ""
    confidence: float = 0.5


@dataclass
class TransitionEntry:
    action: str = "other"
    expected_state_delta: str = ""
    confidence: float = 0.5
    patient_utterance_snippet: str = ""
    therapist_utterance_snippet: str = ""


def _take_recent(seq: List[Any], max_n: int) -> List[Any]:
    if not seq:
        return []
    if max_n is None or max_n <= 0:
        return list(seq)
    return list(seq[-max_n:])


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL)
    if fence:
        s = fence.group(1)
    else:
        i, j = s.find("{"), s.rfind("}")
        if i >= 0 and j > i:
            s = s[i : j + 1]
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def _persona_cot_snippet(raw: str, section: int, max_len: int = 480) -> str:
    """Extract the `### n.` section from the short-term / long-term CoT text, for logging key reasoning."""
    if not raw:
        return ""
    m = re.search(
        rf"###\s*{section}\.\s*[^\n]*\n(.*?)(?=\n###\s*\d+\.|\nMACHINE_JSON:|\Z)",
        raw,
        re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return ""
    s = re.sub(r"\s+", " ", m.group(1).strip())
    return (s[:max_len] + "…") if len(s) > max_len else s


def _optional_therapist_action_str(therapist_action: Optional[str]) -> Optional[str]:
    """Externally supplied discrete action: only strip is applied; vocabulary validation is the caller's responsibility (DataLoader / MSC, etc.)."""
    if therapist_action is None:
        return None
    s = str(therapist_action).strip()
    return s if s else None


class Personas:
    def __init__(self, client, logger, args):
        self.args = args
        self.logger = logger
        self.LLMclient = client
        self._concise_static_logging = bool(getattr(args, "concise_static_logging", True))

        # Latent state space used internally by Personas / Planner (can be decoupled from the evaluation label space).
        persona_space = getattr(args, "persona_state_space", None) or getattr(
            args, "state_label_space", "mi_talk"
        )
        self._state_space = persona_space if persona_space in ("mi_talk", "cami_stage") else "mi_talk"
        if self._state_space == "cami_stage":
            self._patient_talk_types = CAMI_STAGE_TYPES
        else:
            self._patient_talk_types = PATIENT_CLIENT_TALK_TYPES
        self._latent_default_label = self._patient_talk_types[0]

        self.usr_name = getattr(args, "usr_name", "client")
        self.agent_name = getattr(args, "agent_name", "therapist")
        self.max_observation_entries = getattr(args, "max_user_personas", 0)
        self.max_transition_entries = getattr(args, "max_agent_personas", 0)
        self._slow_obs_model_cap = max(400, int(getattr(args, "persona_obs_model_max_chars", 2400) or 2400))
        self._slow_trans_model_cap = max(400, int(getattr(args, "persona_trans_model_max_chars", 2400) or 2400))
        self._q_inference_cap = max(800, int(getattr(args, "persona_q_inference_max_chars", 12000) or 12000))
        self._short_term_prior_prompt_cap = max(
            400, int(getattr(args, "persona_short_term_prior_max_chars", 8000) or 8000)
        )
        self._short_term_window_turns = max(1, int(getattr(args, "persona_short_term_window_turns", 4) or 4))
        self._long_term_summary_turns = max(1, int(getattr(args, "persona_long_term_summary_turns", 12) or 12))
        self._therapist_actions = therapist_action_vocab(
            getattr(args, "therapist_action_granularity", "coarse")
        )

        self._observations: List[ObservationEntry] = []
        self._transitions: List[TransitionEntry] = []
        # The field name is uniformly `client_talk_type`; the value set is controlled by `persona_state_space`.
        self._latent_mi: Dict[str, str] = {"client_talk_type": self._latent_default_label}

        # Two-step inference: three structured state blocks, rendered to JSON text only at prompt time.
        self._q_st_text: List[Dict[str, str]] = []
        self._prev_therapist_utt: str = ""
        self._prev_therapist_action: str = "other"
        self._observation_model_block: List[Dict[str, str]] = []
        self._transition_model_block: List[Dict[str, str]] = self._build_cold_start_transition_rules()
        self._history_buffer: List[Dict[str, Any]] = []
        self._last_inference_entropy: float = 0.5
        # Updated at the end of `_apply_payload`; before the next turn's short-term LLM call inside `traits_update`, this equals the previous turn's short-term 3-state estimate.
        self._q_turn_state_probs: Dict[str, float] = {
            k: 1.0 / len(self._patient_talk_types) for k in self._patient_talk_types
        }
        # Planner's next-state prediction from previous turn → used as transition prior
        self._planner_prior: Optional[Dict[str, float]] = None

    def set_planner_prior(self, predicted_next_label: Optional[str]) -> None:
        """Store planner's next-state prediction as transition prior for next turn."""
        if predicted_next_label and predicted_next_label in self._patient_talk_types:
            # Convert label to soft distribution (peak=0.7, rest split equally)
            probs = {k: 0.15 for k in self._patient_talk_types}
            probs[predicted_next_label] = 0.7
            self._planner_prior = probs
        else:
            self._planner_prior = None

    def _persona_log_info(self, msg: str, *args) -> None:
        if not self.logger:
            return
        if self._concise_static_logging:
            self.logger.debug(msg, *args)
        else:
            self.logger.info(msg, *args)

    def _retry_invalid_machine_json(
        self,
        *,
        raw_bad: str,
        sys_prompt: str,
        user_prompt: str,
        slot_name: str,
        attempt_idx: int = 0,
        total_attempts: int = 1,
    ) -> str:
        if total_attempts <= 0:
            return raw_bad
        reminder = (
            "\n\n---\n"
            "Your previous reply was invalid because it was not a complete parseable JSON object.\n"
            "Return ONLY one valid JSON object.\n"
            "Do NOT output markdown fences.\n"
            "Do NOT output <think> blocks.\n"
            "Do NOT output any prose before or after the JSON.\n"
            "Use the minimal valid schema if uncertain.\n"
            "Invalid previous reply excerpt (do not repeat it):\n"
            f"{(raw_bad or '')[:600]}"
        )
        if self.logger:
            self.logger.warning(
                "[Persona] %s invalid JSON; retry %d/%d with strict reminder.",
                slot_name,
                attempt_idx + 1,
                total_attempts,
            )
        return self.LLMclient.employ(sys_prompt, user_prompt + reminder, "PersonaExtraction")

    def _render_q_st_text(self) -> str:
        return render_q_abstract_rows(self._q_st_text)

    def _render_observation_model_block(self) -> str:
        return render_observation_rules(self._observation_model_block)

    def _render_transition_model_block(self) -> str:
        return render_transition_rules(self._transition_model_block)

    def _render_short_term_memory_window(self) -> str:
        recent = self._history_buffer[-self._short_term_window_turns :]
        if not recent:
            return "[]"
        rows: List[Dict[str, str]] = []
        for idx, item in enumerate(recent, start=1):
            rows.append(
                {
                    "window_turn": str(idx),
                    "a_prev_label": str(item.get("a_prev_label", "") or ""),
                    "a_prev_utt_excerpt": str(item.get("a_prev_utt_excerpt", "") or "")[:160],
                    "patient_o_t_excerpt": str(item.get("o_t_excerpt", "") or "")[:160],
                    "posterior_state": str(item.get("posterior_state", "") or ""),
                    "q_s_post_excerpt": str(item.get("q_s_post_excerpt", "") or "")[:220],
                }
            )
        return json.dumps(rows, ensure_ascii=False, indent=2)

    def _render_raw_dialogue_window(self) -> str:
        """Raw dialogue only — no FEP posterior states, for observation-focused state head.

        Uses obs_state_head_window_turns (default: all available history).
        """
        max_turns = int(getattr(self.args, "obs_state_head_window_turns", 0) or 0)
        if max_turns > 0:
            recent = self._history_buffer[-max_turns:]
        else:
            recent = list(self._history_buffer)  # all history
        if not recent:
            return "(no prior dialogue)"
        lines: List[str] = []
        for item in recent:
            t_utt = str(item.get("a_prev_utt_excerpt", "") or "").strip()
            p_utt = str(item.get("o_t_excerpt", "") or "").strip()
            if t_utt:
                lines.append(f"Therapist: {t_utt}")
            if p_utt:
                lines.append(f"Patient: {p_utt}")
        text = "\n".join(lines) if lines else "(no prior dialogue)"
        # Cap to avoid exceeding context window
        max_chars = 8000
        if len(text) > max_chars:
            text = "(history truncated)\n" + text[-max_chars:]
        return text

    def _build_full_history_text(self, patient_utt: str, therapist_utt: str = "") -> str:
        """Build full dialogue history text (same format as long_prompt baseline)."""
        history_lines: list = []
        for item in self._history_buffer:
            t_utt = str(item.get("a_prev_utt_excerpt", "") or "").strip()
            p_utt = str(item.get("o_t_excerpt", "") or "").strip()
            if t_utt:
                history_lines.append(f"Doctor: {t_utt}")
            if p_utt:
                history_lines.append(f"Patient: {p_utt}")
        if patient_utt:
            history_lines.append(f"Patient: {patient_utt}")
        if therapist_utt:
            history_lines.append(f"Doctor: {therapist_utt}")
        max_chars = int(getattr(self.args, "baseline_history_max_chars", 0) or 0) or 30000
        history_text = "\n".join(history_lines)
        if len(history_text) > max_chars:
            history_text = history_text[-max_chars:]
        return history_text

    def _parse_state_label(self, raw: str) -> Optional[str]:
        """Extract a state label from LLM response JSON."""
        raw_text = (raw or "").strip()
        i = raw_text.find("{")
        j = raw_text.rfind("}")
        if i < 0 or j <= i:
            return None
        try:
            data = json.loads(raw_text[i : j + 1])
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        label = str(data.get("current_state", "")).strip().lower()
        return label if label in CAMI_STAGE_TYPES else None

    def _observation_focused_state_call(self, patient_utt: str, therapist_utt: str = "") -> Optional[Dict[str, float]]:
        """Two-step state inference:
        Step 1: Long-prompt style classification (full history, current_state only)
        Step 2: Refinement — give LLM its Step 1 prediction + the prior from last turn,
                ask it to decide whether to keep or revise.
        """
        if tuple(self._patient_talk_types) != CAMI_STAGE_TYPES:
            return None

        history_text = self._build_full_history_text(patient_utt, therapist_utt)

        # ---- Step 1: Pure long-prompt classification ----
        sys_prompt_1 = (
            "You are a readiness stage classifier using the Transtheoretical Model. "
            "Infer the patient's current readiness stage based on their latest utterance and the interaction history. "
            "Use ONLY these labels: precontemplation, contemplation, preparation.\n"
            "- precontemplation: the patient does not recognize the behavior as problematic and tends to defend the status quo.\n"
            "- contemplation: the patient recognizes some problem signals but remains ambivalent and mixed about change.\n"
            "- preparation: the patient is ready to take action and expresses concrete movement toward change.\n"
            "Return strict JSON."
        )
        user_prompt_1 = (
            "Given the full interaction history, infer the patient's current readiness stage.\n\n"
            "Output JSON exactly as:\n"
            '{"current_state":"precontemplation|contemplation|preparation","rationale":"<brief>"}\n\n'
            f"Full history:\n{history_text}"
        )
        raw_1 = self.LLMclient.employ(sys_prompt_1, user_prompt_1, "SemanticStateHead_Step1")
        step1_label = self._parse_state_label(raw_1)

        if self.logger:
            self._persona_log_info(
                "[Persona] obs_step1: long_prompt_label=%s", step1_label
            )

        if step1_label is None:
            return None

        # ---- No Step 2: use Step1 result directly ----
        # Planner prior has been shown to introduce contemplation bias
        # (transition rules learned on biased labels → planner always predicts
        # contemplation → preparation recall collapses). Pure long-prompt is better.
        if self.logger:
            self._persona_log_info(
                "[Persona] obs_state: using step1=%s directly (no prior refinement)",
                step1_label,
            )
        probs = {k: 0.05 for k in CAMI_STAGE_TYPES}
        probs[step1_label] = 0.9
        return probs

    def _compute_transition_prior(self) -> Dict[str, float]:
        """Compute transition prior p̃(s_t) from transition model rules + previous state/action."""
        prev_state = self._latent_mi.get("client_talk_type", self._latent_default_label)
        prev_action = self._prev_therapist_action or "other"
        # Count matching transition rules
        counts: Dict[str, float] = {label: 0.0 for label in self._patient_talk_types}
        for rule in self._transition_model_block:
            s_from = str(rule.get("s_from", "")).strip().lower()
            action = str(rule.get("action_a", "")).strip().lower()
            s_to = str(rule.get("s_to", "")).strip().lower()
            if s_from == prev_state and s_to in counts:
                # Exact action match gets full weight, any-action gets half
                weight = 1.0 if action == prev_action else 0.3
                counts[s_to] += weight
        # Laplace smoothing + normalize
        n = len(self._patient_talk_types)
        total = sum(counts.values()) + n
        return {k: (v + 1.0) / total for k, v in counts.items()}

    @staticmethod
    def _bayesian_combine(
        prior: Dict[str, float],
        likelihood: Dict[str, float],
    ) -> Dict[str, float]:
        """q(s_t) ∝ prior × likelihood"""
        raw = {k: prior.get(k, 1e-6) * likelihood.get(k, 1e-6) for k in prior}
        total = sum(raw.values())
        if total < 1e-12:
            n = len(raw)
            return {k: 1.0 / n for k in raw}
        return {k: v / total for k, v in raw.items()}

    def _render_long_term_memory_summary(self) -> str:
        recent = self._history_buffer[-self._long_term_summary_turns :]
        stage_trace = [
            str(item.get("posterior_state", "") or "").strip()
            for item in recent
            if str(item.get("posterior_state", "") or "").strip()
        ]
        payload = {
            "current_latent_state": self._latent_mi.get("client_talk_type", self._latent_default_label),
            "recent_state_trace": stage_trace[-self._long_term_summary_turns :],
            "observation_rule_count": len(self._observation_model_block),
            "transition_rule_count": len(self._transition_model_block),
            "recent_observation_rules": self._observation_model_block[-4:],
            "recent_transition_rules": self._transition_model_block[-4:],
            "recent_interaction_summaries": [
                {
                    "a_prev_label": str(item.get("a_prev_label", "") or ""),
                    "patient_o_t_excerpt": str(item.get("o_t_excerpt", "") or "")[:120],
                    "posterior_state": str(item.get("posterior_state", "") or ""),
                }
                for item in recent[-6:]
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def traits_update(
        self,
        patient_utterance: str,
        therapist_utterance: str,
        therapist_action: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """
        One-step update for the current dialogue turn (aligned with the "P_i then D_i" order used by MSC).

        :param therapist_action: Optional. If supplied, the stripped value is used as the **current-turn** therapist discrete action a_t,
            written into the long-term prompt and overriding the LLM-returned therapist_action; if not supplied, the model still produces the annotation.
            Vocabulary consistency is enforced by the caller (e.g. MSC).

        Default: (1) short-term rolling `_q_st_text` (per-turn revision on top of the previous narrative) + `latent_mi`;
        (2) long-term p(o|s) and p(s'|s,a) text blocks incrementally merged (slow-changing) + current therapist-utterance label / transition.
        If `args.persona_single_step` is set: a single legacy-JSON call.
        """
        action_override = _optional_therapist_action_str(therapist_action)
        self._persona_log_info(
            "[Persona] traits_update start mode=%s patient_chars=%d therapist_chars=%d "
            "prior_q_chars=%d a_prev=%s action_override=%s",
            "legacy_single" if getattr(self.args, "persona_single_step", False) else "two_step_fep",
            len(patient_utterance or ""),
            len(therapist_utterance or ""),
            len(self._render_q_st_text()),
            self._prev_therapist_action,
            action_override or "(none)",
        )
        if getattr(self.args, "persona_single_step", False):
            payload = self._infer_legacy_single_json(
                patient_utterance, therapist_utterance, action_override=action_override
            )
        else:
            payload = self._infer_two_step_fep(
                patient_utterance, therapist_utterance, action_override=action_override
            )
        self._apply_payload(payload, patient_utterance, therapist_utterance)
        self._persona_log_info(
            "[Persona] traits_update done client_talk_type=%s n_observations=%d n_transitions=%d",
            self._latent_mi.get("client_talk_type", "?"),
            len(self._observations),
            len(self._transitions),
        )
        return (
            self._format_current_state(),
            self._format_observation_model(),
            self._format_transition_model(),
        )

    def replay_two_step_from_raw_outputs(
        self,
        raw_short: str,
        raw_long: str,
        *,
        patient_utt: str,
        therapist_utt: str,
        therapist_action: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Re-parse the stored short-/long-term raw text and merge it back into the current Personas without calling the LLM (used for log replay, unit tests, and debugging).
        Uses the same `parse_*` + `apply_two_step_machine_json_to_personas` as `_infer_two_step_fep`,
        and immediately calls `_apply_payload`, matching the side effects of `traits_update` under the two-step mode.
        """
        prior_q_full = self._render_q_st_text().strip()
        max_h = max(1, int(getattr(self.args, "persona_history_max", 6) or 6))
        action_override = _optional_therapist_action_str(therapist_action)
        st = parse_short_term_llm_output(
            raw_short or "",
            patient_utt=patient_utt,
            prior_q_full=prior_q_full,
            fallback_client_talk_type=self._latent_mi.get("client_talk_type", "neutral"),
            patient_talk_types=PATIENT_CLIENT_TALK_TYPES,
            logger=self.logger,
        )
        lt = parse_long_term_llm_output(
            raw_long or "",
            therapist_actions=self._therapist_actions,
            action_override=action_override,
            logger=self.logger,
        )
        payload = apply_two_step_machine_json_to_personas(
            self,
            st,
            lt,
            patient_utt=patient_utt,
            therapist_utt=therapist_utt,
            prior_q_full=prior_q_full,
            max_history=max_h,
            logger=self.logger,
        )
        self._apply_payload(payload, patient_utt, therapist_utt)
        return payload

    def _infer_legacy_single_json(
        self,
        patient_utt: str,
        therapist_utt: str,
        action_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        user_prompt = format_persona_extraction_user(
            self.usr_name,
            self.agent_name,
            patient_utt,
            therapist_utt,
            json.dumps(self._latent_mi, ensure_ascii=False),
            MI_LATENT_KEYS,
            self._therapist_actions,
            self._patient_talk_types,
        )
        sys_prompt = build_persona_extraction_sys(self._patient_talk_types)
        raw = self.LLMclient.employ(sys_prompt, user_prompt, "PersonaExtraction")
        if self.logger:
            self.logger.debug(
                "[Persona] legacy raw_chars=%d raw_tail=%s",
                len(raw or ""),
                ((raw or "")[-400:].replace("\n", " ") if raw else ""),
            )
        parsed = _extract_json_object(raw)
        if parsed is None:
            if self.logger:
                self.logger.warning(
                    "[Persona] legacy PersonaExtraction JSON parse failed; using fallback."
                )
            return self._fallback_payload(patient_utt, therapist_utt)
        if self.logger:
            lm = parsed.get("latent_mi") or {}
            ct = lm.get("client_talk_type", "?") if isinstance(lm, dict) else "?"
            self._persona_log_info(
                "[Persona] legacy parsed client_talk_type=%s therapist_action=%s",
                ct,
                parsed.get("therapist_action", "?"),
            )
        if action_override:
            parsed["therapist_action"] = action_override
        return parsed

    def _infer_two_step_fep(
        self,
        patient_utt: str,
        therapist_utt: str,
        action_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        # The short-term belief accumulated before this turn's update (full text stored in `_q_st_text`); the prompt uses a truncated version to control tokens.
        prior_q_full = self._render_q_st_text().strip()
        _cap = self._short_term_prior_prompt_cap
        prior_for_prompt = prior_q_full
        if _cap > 0 and len(prior_for_prompt) > _cap:
            prior_for_prompt = "(prior truncated)\n" + prior_for_prompt[-_cap:]

        # --- Skip q filter when observation_focused_state is enabled ---
        # The two-step LLM call (Step1 long-prompt + Step2 prior refinement)
        # in _observation_focused_state_call replaces q filter's state classification.
        # We only need p filter for observation/transition rule updates.
        skip_q_filter = bool(
            getattr(self.args, "observation_focused_state", False)
        )

        if skip_q_filter:
            # State is already set by MSC (long-prompt + optional prior refinement)
            # via personas._latent_mi["client_talk_type"] BEFORE calling traits_update.
            # Do NOT call _observation_focused_state_call here — it would overwrite
            # the correct state with a worse prediction (truncated history).
            # Just provide a stub for p filter.
            current_label = self._latent_mi.get("client_talk_type", self._latent_default_label)
            self._q_turn_state_probs = {current_label: 0.9}
            q_text = (f"Current patient readiness stage: {current_label}. "
                      f"Patient said: {(patient_utt or '')[:200]}")
            st = type("_StubST", (), {
                "client_talk_type": current_label,
                "inference_entropy": 0.0,
                "q_manifold": q_text,
                "machine_json": {"client_talk_type": current_label, "inference_entropy": 0.0},
                "observation": {"state_pattern": q_text, "confidence": 0.5},
                "q_delta_normalized": {},
                "turn_body_for_q_roll": "",
            })()
            if self.logger:
                self._persona_log_info(
                    "[Persona] short_term_q SKIPPED, state=%s set by two-step LLM (p filter will use this)",
                    current_label,
                )
        else:
            if self.logger:
                self.logger.debug(
                    "[Persona] two_step prior_stored_chars=%d prior_prompt_chars=%d obs_block_chars=%d trans_block_chars=%d",
                    len(prior_q_full),
                    len(prior_for_prompt),
                    len(self._render_observation_model_block()),
                    len(self._render_transition_model_block()),
                )

            st_user = format_persona_short_term_q_user(
                prior_for_prompt,
                self._prev_therapist_action,
                self._prev_therapist_utt,
                self._render_transition_model_block(),
                self._render_observation_model_block(),
                self._render_short_term_memory_window(),
                patient_utt,
                self._patient_talk_types,
            )
            raw_st = self.LLMclient.employ(PERSONA_SHORT_TERM_Q_SYS, st_user, "PersonaExtraction")
            if self.logger:
                q_head = (raw_st or "")[:800].replace("\n", " ")
                self._persona_log_info(
                    "[Persona] short_term_q_raw len=%d head=%s%s",
                    len(raw_st or ""),
                    q_head,
                    raw_st
                )
                self.logger.debug("[Persona] short_term_q raw: %s", raw_st)
                s1 = _persona_cot_snippet(raw_st or "", 1)
                s2 = _persona_cot_snippet(raw_st or "", 2)
                if s1:
                    self.logger.debug("[Persona] short_term_q §1 Prediction_Error: %s", s1)
                if s2:
                    self.logger.debug("[Persona] short_term_q §2 Delta_q: %s", s2)
            st = parse_short_term_llm_output(
                raw_st or "",
                patient_utt=patient_utt,
                prior_q_full=prior_q_full,
                fallback_client_talk_type=self._latent_mi.get("client_talk_type", self._latent_default_label),
                patient_talk_types=tuple(self._patient_talk_types),
                logger=self.logger,
            )
            if not st.machine_json:
                retry_max = max(0, int(getattr(self.args, "persona_json_retry_max", 1) or 1))
                for ridx in range(retry_max):
                    raw_st = self._retry_invalid_machine_json(
                        raw_bad=raw_st or "",
                        sys_prompt=PERSONA_SHORT_TERM_Q_SYS,
                        user_prompt=st_user,
                        slot_name="short_term_q",
                        attempt_idx=ridx,
                        total_attempts=retry_max,
                    )
                    st = parse_short_term_llm_output(
                        raw_st or "",
                        patient_utt=patient_utt,
                        prior_q_full=prior_q_full,
                        fallback_client_talk_type=self._latent_mi.get("client_talk_type", self._latent_default_label),
                        patient_talk_types=tuple(self._patient_talk_types),
                        logger=self.logger,
                    )
                    if st.machine_json:
                        break
            if self.logger and st.machine_json:
                self._persona_log_info(
                    "[Persona] short_term_q MACHINE_JSON client_talk_type=%s inference_entropy=%s",
                    st.machine_json.get("client_talk_type", "(missing)"),
                    st.machine_json.get("inference_entropy", "(missing)"),
                )
            self._persona_log_info(
                "[Persona] short_term_q resolved client_talk_type=%s entropy=%.3f q_manifold_chars=%d excerpt=%s",
                st.client_talk_type,
                st.inference_entropy,
                len(st.q_manifold),
                (st.q_manifold[:320].replace("\n", " ") + "…")
                if len(st.q_manifold) > 320
                else st.q_manifold.replace("\n", " "),
            )

        max_h = max(1, int(getattr(self.args, "persona_history_max", 6) or 6))
        history_json = json.dumps(self._history_buffer[-max_h:], ensure_ascii=False)

        lt_user = format_persona_long_term_p_user(
            self._render_long_term_memory_summary()[:2200],
            history_json[:1800],
            self._render_observation_model_block()[:1200],
            self._render_transition_model_block()[:1200],
            prior_q_full[:700],
            self._prev_therapist_action,
            self._prev_therapist_utt,
            st.q_manifold,
            patient_utt,
            therapist_utt,
            ", ".join(self._therapist_actions),
            self._patient_talk_types,
            given_therapist_action=action_override,
        )
        raw_lt = self.LLMclient.employ(PERSONA_LONG_TERM_P_SYS, lt_user, "PersonaExtraction")
        if self.logger:
            # p_raw: raw long-term p(o|s) and p(s'|s,a) output (truncated).
            p_head = (raw_lt or "")[:800].replace("\n", " ")
            self._persona_log_info(
                "[Persona] long_term_p_raw len=%d head=%s%s",
                len(raw_lt or ""),
                p_head,
                raw_lt
            )
            self.logger.debug("[Persona] long_term_p raw: %s", raw_lt)
            lt_cot = _persona_cot_snippet(raw_lt or "", 1)
            if lt_cot:
                self.logger.debug("[Persona] long_term_p §1 Free_energy_CoT: %s", lt_cot)
            else:
                head = (raw_lt or "")[:500].replace("\n", " ")
                if head:
                    self.logger.debug("[Persona] long_term_p prose_head: %s…", head[:480])

        lt = parse_long_term_llm_output(
            raw_lt or "",
            therapist_actions=self._therapist_actions,
            action_override=action_override,
            logger=self.logger,
        )
        if not lt.machine_json:
            retry_max = max(0, int(getattr(self.args, "persona_json_retry_max", 1) or 1))
            for ridx in range(retry_max):
                raw_lt = self._retry_invalid_machine_json(
                    raw_bad=raw_lt or "",
                    sys_prompt=PERSONA_LONG_TERM_P_SYS,
                    user_prompt=lt_user,
                    slot_name="long_term_p",
                    attempt_idx=ridx,
                    total_attempts=retry_max,
                )
                lt = parse_long_term_llm_output(
                    raw_lt or "",
                    therapist_actions=self._therapist_actions,
                    action_override=action_override,
                    logger=self.logger,
                )
                if lt.machine_json:
                    break

        return apply_two_step_machine_json_to_personas(
            self,
            st,
            lt,
            patient_utt=patient_utt,
            therapist_utt=therapist_utt,
            prior_q_full=prior_q_full,
            max_history=max_h,
            logger=self.logger,
        )

    def _fallback_payload(self, patient_utt: str, therapist_utt: str) -> Dict[str, Any]:
        if self.logger:
            self.logger.warning(
                "[Persona] fallback_payload: weak snapshot patient_excerpt=%s",
                ((patient_utt or "")[:100].replace("\n", " ") + "…")
                if len(patient_utt or "") > 100
                else (patient_utt or "")[:100],
            )
        snippet_p = (patient_utt or "")[:120].replace("\n", " ")
        snippet_t = (therapist_utt or "")[:120].replace("\n", " ")
        return {
            "observation": {
                "state_pattern": "Unparsed LLM output; retain raw utterance as weak evidence.",
                "textual_cues": snippet_p or "(empty)",
                "behavioral_cues": "(not inferred)",
                "confidence": 0.2,
            },
            "latent_mi": dict(self._latent_mi),
            "therapist_action": "other",
            "transition": {
                "expected_state_delta": f"Therapist move (truncated): {snippet_t}",
                "confidence": 0.2,
            },
        }

    def _apply_payload(
        self,
        data: Dict[str, Any],
        patient_utt: str,
        therapist_utt: str,
    ) -> None:
        self._apply_observation_and_latent_from_patient(data)
        self._apply_transition_from_therapist_action(data, patient_utt, therapist_utt)

        # --- Observation-focused state override ---
        # When observation_focused_state=True, the two-step classification
        # (Step1 long-prompt + Step2 prior refinement) already ran in
        # _infer_two_step_fep BEFORE p filter. State is already set.
        # No action needed here.

        if self.logger:
            obs = data.get("observation") or {}
            full_sp = str(obs.get("state_pattern", ""))
            sp_excerpt = full_sp[:200].replace("\n", " ")
            if len(full_sp) > 200:
                sp_excerpt += "…"
            self._persona_log_info(
                "[Persona] apply_payload observation_pattern_excerpt=%s appended_transition_action=%s",
                sp_excerpt,
                str(data.get("therapist_action", "")),
            )

    def _apply_observation_and_latent_from_patient(self, data: Dict[str, Any]) -> None:
        """p(o_t|s_t): induce the observable cues o_t from the patient utterance, and make a small update to the `latent_mi` belief about s_t."""
        obs = data.get("observation") or {}
        entry = ObservationEntry(
            state_pattern=str(obs.get("state_pattern", "")).strip(),
            textual_cues=str(obs.get("textual_cues", "")).strip(),
            behavioral_cues=str(obs.get("behavioral_cues", "")).strip(),
            prediction_error=str(obs.get("prediction_error", "")).strip(),
            confidence=clamp_conf(obs.get("confidence")),
        )
        if any(
            [
                entry.state_pattern,
                entry.textual_cues,
                entry.behavioral_cues,
                entry.prediction_error,
            ]
        ):
            self._observations.append(entry)

        lm = data.get("latent_mi") or {}
        if isinstance(lm, dict):
            v = lm.get("client_talk_type")
            if v is not None and str(v).strip():
                vv = str(v).strip().lower()
                if vv in self._patient_talk_types:
                    self._latent_mi["client_talk_type"] = vv

        # The 3-state q(s) distribution at the end of this turn's short-term inference; until the next turn's inference begins, this serves as the "previous-turn short-term estimate".
        self._q_turn_state_probs = estimate_q_turn_state_probs_from_payload(
            data, patient_talk_types=tuple(self._patient_talk_types)
        )

    def _apply_transition_from_therapist_action(
        self,
        data: Dict[str, Any],
        patient_utt: str,
        therapist_utt: str,
    ) -> None:
        """p(s_t|s_{t-1}, a_{t-1}): therapist action a together with the expected latent-state change."""
        action = str(data.get("therapist_action", "other")).strip()
        if not action:
            action = "other"

        tr = data.get("transition") or {}
        self._transitions.append(
            TransitionEntry(
                action=action,
                expected_state_delta=str(
                    tr.get("therapist_action_rationale") or tr.get("expected_state_delta", "")
                ).strip(),
                confidence=clamp_conf(tr.get("confidence")),
                patient_utterance_snippet=(patient_utt or "")[:80],
                therapist_utterance_snippet=(therapist_utt or "")[:80],
            )
        )

    def _format_current_state(self) -> str:
        return json.dumps(
            {"client_talk_type": self._latent_mi.get("client_talk_type", "neutral")},
            ensure_ascii=False,
            indent=2,
        )

    def _format_observation_model(self) -> str:
        return self._render_observation_model_block() or "[]"

    def _format_transition_model(self) -> str:
        return self._render_transition_model_block() or "[]"

    def _format_user_traits(self) -> str:
        """Keep a compact user model, for debugging or backward-compatible callers."""
        lines = [
            "### Current patient state",
            self._format_current_state(),
            "",
            "### Observation model p(o|s)",
            self._format_observation_model(),
        ]
        return "\n".join(lines)

    def _format_agent_traits(self) -> str:
        """Keep a compact therapist-side model, for debugging or backward-compatible callers."""
        lines = [
            "### Transition model p(s'|s,a)",
            self._format_transition_model(),
        ]
        return "\n".join(lines)

    def _build_cold_start_transition_rules(self) -> List[Dict[str, str]]:
        """
        Build uniform cold-start transition rules — NO data leakage.

        All actions start with the same low-confidence prior: "might keep
        current state". This ensures EFE treats all unknown actions equally
        and relies on the adaptive λ_e/λ_p to drive exploration of untried
        actions. As the Personas module observes actual transitions, learned
        rules will override these priors.
        """
        if len(self._therapist_actions) <= 4:
            return []  # Only for fine-grained vocab

        return []  # Start empty — let EFE's optimistic exploration prior handle unknown actions

    @property
    def therapist_actions(self) -> Tuple[str, ...]:
        """The discrete therapist-action vocabulary for the current run (determined by `therapist_action_granularity`)."""
        return self._therapist_actions

    # --- Optional: for debugging or downstream modules to read the structured state ---
    @property
    def q_turn_state_probs(self) -> Dict[str, float]:
        """A copy of the previous-turn short-term 3-state distribution after it was committed (refreshed at the end of every `_apply_payload`)."""
        return dict(self._q_turn_state_probs)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "latent_mi": dict(self._latent_mi),
            "observations": [e.__dict__ for e in self._observations],
            "transitions": [e.__dict__ for e in self._transitions],
            "q_st_text": [dict(x) for x in self._q_st_text],
            "last_inference_entropy": self._last_inference_entropy,
            "q_turn_state_probs": dict(self._q_turn_state_probs),
            "observation_model_block": [dict(x) for x in self._observation_model_block],
            "transition_model_block": [dict(x) for x in self._transition_model_block],
            "persona_history": list(self._history_buffer),
        }

    def format_full_user_model(self) -> str:
        """Compact combined view: current state + observation model + transition model."""
        return "\n\n".join(
            [
                "### Current patient state",
                self._format_current_state(),
                "### Observation model p(o|s)",
                self._format_observation_model(),
                "### Transition model p(s'|s,a)",
                self._format_transition_model(),
            ]
        )

    def last_therapist_action(self) -> str:
        if not self._transitions:
            return "other"
        a = (self._transitions[-1].action or "").strip()
        return a if a else "other"
