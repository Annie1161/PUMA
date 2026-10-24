"""
Action-conditioned transition planner: given the current persona belief, the user model
(observation + transition tracks) and the therapist action a, estimate the patient's
latent state and behavioral tendency after that action, for the client-reply generator.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Sequence

from Module.prompts import build_planner_sys, format_planner_user

from Module.Personas import MI_LATENT_KEYS, CAMI_STAGE_TYPES, PATIENT_CLIENT_TALK_TYPES
from tools.persona_machine_json import normalize_therapist_action


def _strip_model_noise(text: str) -> str:
    s = text.strip()
    s = re.sub(r"<redacted_thinking>.*?</redacted_thinking>", "", s, flags=re.DOTALL | re.IGNORECASE)
    return s.strip()


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """First balanced JSON object via raw_decode (avoids greedy brace slice breaking nested strings)."""
    if not text:
        return None
    s = _strip_model_noise(text)
    fm = re.search(r"```(?:json)?\s*", s, re.IGNORECASE)
    if fm:
        rest = s[fm.end() :]
        end_fence = rest.find("```")
        if end_fence >= 0:
            rest = rest[:end_fence]
        jb = rest.find("{")
        if jb >= 0:
            try:
                obj, _ = json.JSONDecoder().raw_decode(rest[jb:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
    i = s.find("{")
    while i >= 0:
        try:
            obj, _ = json.JSONDecoder().raw_decode(s[i:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        i = s.find("{", i + 1)
    return None


def _format_planner_text(data: Dict[str, Any], state_labels: Sequence[str]) -> str:
    lines = [
        "### Roll-forward: patient state after therapist action (under p(s'|s, a))",
        f"(model confidence: {float(data.get('confidence', 0.5)):.2f})",
        "",
        "#### Next latent: client_talk_type (" + " / ".join(state_labels) + ")",
    ]
    nxt = data.get("latent_mi_next") or {}
    if isinstance(nxt, dict):
        for k in MI_LATENT_KEYS:
            lines.append(f"- {k}: {nxt.get(k, 'unknown')}")
    else:
        for k in MI_LATENT_KEYS:
            lines.append(f"- {k}: unknown")

    lines.append("")
    lines.append("#### Expected next verbal / behavioral cues (emission tendency)")
    lines.append(str(data.get("expected_behavior_cues", "")).strip() or "(unspecified)")
    lines.append("")
    lines.append("#### Brief rationale")
    lines.append(str(data.get("rationale", "")).strip() or "(none)")
    return "\n".join(lines)


_PREPARATION_CUE_SNIPPETS = (
    "what do you think that you need to do",
    "what do you need to do",
    "next, like, month",
    "next month",
    "next week",
    "first step",
    "realistic goal",
    "how long do you think you could go",
    "what things would make it easier",
    "who can you talk to",
    "what could you do to avoid",
    "plan",
)


def _looks_like_preparation_evoke(therapist_utterance: str) -> bool:
    text = (therapist_utterance or "").strip().lower()
    if not text:
        return False
    return any(sn in text for sn in _PREPARATION_CUE_SNIPPETS)


def _calibrate_cami_next_state(
    data: Dict[str, Any],
    *,
    therapist_utterance: str,
) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return data
    nxt = data.get("latent_mi_next")
    if not isinstance(nxt, dict):
        return data
    pred = str(nxt.get("client_talk_type", "") or "").strip().lower()
    if pred not in CAMI_STAGE_TYPES:
        return data

    plan_cue = _looks_like_preparation_evoke(therapist_utterance)

    if not (pred == "contemplation" and plan_cue):
        return data

    out = dict(data)
    latent = dict(nxt)
    latent["client_talk_type"] = "preparation"
    out["latent_mi_next"] = latent
    try:
        conf = float(out.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    out["confidence"] = max(conf, 0.55)
    return out


class TransitionPlanner:
    def __init__(self, client, logger, args):
        self.LLMclient = client
        self.logger = logger
        self.args = args
        persona_space = getattr(args, "persona_state_space", None) or getattr(
            args, "state_label_space", "mi_talk"
        )
        if persona_space == "cami_stage":
            self._state_labels = CAMI_STAGE_TYPES
        else:
            self._state_labels = PATIENT_CLIENT_TALK_TYPES

    def predict_post_action_state(
        self,
        current_state: str,
        observation_model: str,
        transition_model: str,
        action: Optional[str],
        therapist_utterance: str,
        therapist_actions: Sequence[str],
    ) -> str:
        """
        Inputs: current patient state, current p(o|s), current p(s'|s,a).
        Action: an explicit label; falls back to "other" when empty.
        Output: a formatted "post-action patient state" text block, injected into the client-reply generator.
        """
        granularity = getattr(self.args, "therapist_action_granularity", "coarse")
        act = normalize_therapist_action(
            action,
            granularity=granularity,
            vocab=tuple(therapist_actions),
            fallback_other=True,
        ) or "other"

        user_prompt = format_planner_user(
            current_state,
            observation_model,
            transition_model,
            act,
            therapist_utterance,
            MI_LATENT_KEYS,
            therapist_actions,
            self._state_labels,
        )
        sys_prompt = build_planner_sys(self._state_labels)
        raw = self.LLMclient.employ(sys_prompt, user_prompt, "Planner")
        if self.logger:
            self.logger.debug("Planner: user next latent state esitmation raw: %s", raw)
        data = _extract_json_object(raw)
        if self.logger:
            self.logger.info("Planner: user next latent state esitmation: %s", data)
        
        if not data:
            if self.logger:
                self.logger.debug("Planner: JSON parse failed; using fallback text.")
            return self._fallback_text(act, therapist_utterance, current_state)
        if (
            tuple(self._state_labels) == CAMI_STAGE_TYPES
            and bool(getattr(self.args, "cami_preparation_calibration", True))
        ):
            data = _calibrate_cami_next_state(
                data,
                therapist_utterance=therapist_utterance,
            )
        return _format_planner_text(data, self._state_labels)

    def _fallback_text(self, action: str, therapist_utterance: str, current_state: str) -> str:
        latent = _extract_json_object(current_state) or {}
        latent = latent if isinstance(latent, dict) else {}
        latent_mi = latent.get("latent_mi") if isinstance(latent.get("latent_mi"), dict) else latent
        t_snip = (therapist_utterance or "")[:160].replace("\n", " ")
        return _format_planner_text(
            {
                "confidence": 0.25,
                "latent_mi_next": dict(latent_mi) if isinstance(latent_mi, dict) else {},
                "expected_behavior_cues": f"Unplanned fallback under action={action}; therapist said (truncated): {t_snip}",
                "rationale": "Planner JSON parse failed; latent sketch copied from pre-action belief.",
            },
            self._state_labels,
        )
