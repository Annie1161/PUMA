"""
MITI Behavior Scores evaluator.

Computes the 5 MITI (Motivational Interviewing Treatment Integrity) Behavior Scores
for a counselor's utterances, following Yang et al. (2025) CAMI paper Section 4.3
and Appendix A.5.

The five scores are:
  R/Q    : Reflection/Question ratio = (SR+CR) / (OQ+CQ)
  %OQ    : Proportion of Open Questions = OQ / (OQ+CQ) * 100
  %CR    : Proportion of Complex Reflections = CR / (SR+CR) * 100
  %MIC   : Proportion of MI-Consistent Responses = MICO / (MICO+MIIN) * 100
  %TTT   : Percent Therapist Talk Time = counselor_words / total_words * 100

Behavior labels follow the MI Skill Code (MISC, Miller et al. 2003):
  open_question, closed_question, simple_reflection, complex_reflection,
  mico (MI-Consistent), miin (MI-Inconsistent), other
"""

import re
import json
import logging
from typing import Optional

BEHAVIOR_LABELS = [
    "open_question",
    "closed_question",
    "simple_reflection",
    "complex_reflection",
    "mico",
    "miin",
    "other",
]

_CLASSIFY_SYS = """\
You are an expert Motivational Interviewing (MI) behavior coder trained on the MI Skill Code (MISC).
Classify the counselor utterance into EXACTLY ONE of the following behavior categories:

- open_question: Open-ended question that invites elaboration (e.g. "What brings you here?", "How do you feel about...?")
- closed_question: Closed/yes-no question or one that elicits a short specific answer
- simple_reflection: Restatement or shallow paraphrase of client content — no new meaning added
- complex_reflection: Reflection that adds meaning, makes an inference, emphasizes change talk, or summarizes multiple themes (paraphrase/summary beyond surface content)
- mico: MI-Consistent behavior — affirms, emphasizes autonomy, supports, raises concern collaboratively, reframes, seeks collaboration, or otherwise aligns with MI spirit
- miin: MI-Inconsistent behavior — confronts, argues, lectures, warns, directs, advises without permission, or undermines client autonomy
- other: Filler, greeting, social talk, or any utterance that does not fit the above

Return ONLY valid JSON: {"label": "<one of the seven categories above>"}"""


def _parse_behavior_label(raw: str) -> str:
    """Extract and validate behavior label from LLM response."""
    if not raw:
        return "other"
    m = re.search(r'"label"\s*:\s*"([^"]+)"', raw)
    if m:
        candidate = m.group(1).strip().lower().replace(" ", "_").replace("-", "_")
        if candidate in BEHAVIOR_LABELS:
            return candidate
    # Try to find any label keyword in the text
    text = raw.lower()
    for label in BEHAVIOR_LABELS:
        if label in text:
            return label
    return "other"


class MITIScorer:
    """
    Accumulates MITI behavior label counts for counselor utterances
    and computes the 5 MITI Behavior Scores at the end of a session.

    Usage:
        scorer = MITIScorer(client, logger)
        for counselor_utt, patient_utt in turns:
            scorer.add_turn(counselor_utt, patient_utt)
        scores = scorer.compute_scores()
        scorer.reset()  # for next sample/session
    """

    def __init__(self, client, logger: Optional[logging.Logger] = None):
        self.client = client
        self.logger = logger
        self.reset()

    def reset(self):
        self.counts = {k: 0 for k in BEHAVIOR_LABELS}
        self.counselor_word_count = 0
        self.patient_word_count = 0

    def classify_utterance(self, utterance: str) -> str:
        """Classify a single counselor utterance into a MISC behavior label via LLM."""
        user_prompt = f'Counselor utterance: "{utterance}"\n\nClassify this utterance.'
        try:
            raw = self.client.employ(_CLASSIFY_SYS, user_prompt, "ResponseGenerator")
            label = _parse_behavior_label(raw)
            if self.logger:
                self.logger.debug(
                    "[MITI] classify utterance=%r -> label=%s (raw=%r)",
                    utterance[:80],
                    label,
                    (raw or "")[:120],
                )
            return label
        except Exception as exc:
            if self.logger:
                self.logger.warning("[MITI] classify_utterance failed: %s", exc)
            return "other"

    def add_turn(self, counselor_utt: str, patient_utt: str) -> str:
        """
        Classify the counselor utterance, update counts and word tallies.
        Returns the predicted behavior label.
        """
        label = self.classify_utterance(counselor_utt)
        self.counts[label] += 1
        self.counselor_word_count += len((counselor_utt or "").split())
        self.patient_word_count += len((patient_utt or "").split())
        return label

    def compute_scores(self) -> dict:
        """
        Compute the 5 MITI Behavior Scores from accumulated counts.
        Returns a dict with keys: R_Q, pct_OQ, pct_CR, pct_MIC, pct_TTT,
        plus raw counts and word tallies.
        """
        n_reflect = self.counts["simple_reflection"] + self.counts["complex_reflection"]
        n_question = self.counts["open_question"] + self.counts["closed_question"]
        n_mico = self.counts["mico"]
        n_miin = self.counts["miin"]
        total_words = self.counselor_word_count + self.patient_word_count

        rq = (n_reflect / n_question) if n_question > 0 else None
        pct_oq = (self.counts["open_question"] / n_question * 100) if n_question > 0 else None
        pct_cr = (self.counts["complex_reflection"] / n_reflect * 100) if n_reflect > 0 else None
        pct_mic = (n_mico / (n_mico + n_miin) * 100) if (n_mico + n_miin) > 0 else None
        pct_ttt = (self.counselor_word_count / total_words * 100) if total_words > 0 else None

        return {
            "R_Q": round(rq, 4) if rq is not None else None,
            "pct_OQ": round(pct_oq, 2) if pct_oq is not None else None,
            "pct_CR": round(pct_cr, 2) if pct_cr is not None else None,
            "pct_MIC": round(pct_mic, 2) if pct_mic is not None else None,
            "pct_TTT": round(pct_ttt, 2) if pct_ttt is not None else None,
            "n_reflect": n_reflect,
            "n_question": n_question,
            "n_mico": n_mico,
            "n_miin": n_miin,
            "counts": dict(self.counts),
            "counselor_words": self.counselor_word_count,
            "patient_words": self.patient_word_count,
            "total_turns_scored": sum(self.counts.values()),
        }


def aggregate_miti_scores(per_sample_scores: list) -> dict:
    """
    Aggregate a list of per-sample MITI score dicts into a corpus-level summary.
    Uses micro-averaging over raw counts (not macro-averaging over ratios).
    """
    if not per_sample_scores:
        return {}

    total_n_reflect = 0
    total_n_question = 0
    total_n_mico = 0
    total_n_miin = 0
    total_oq = 0
    total_cr = 0
    total_counselor_words = 0
    total_patient_words = 0
    total_turns = 0

    for s in per_sample_scores:
        total_n_reflect += s.get("n_reflect", 0)
        total_n_question += s.get("n_question", 0)
        total_n_mico += s.get("n_mico", 0)
        total_n_miin += s.get("n_miin", 0)
        counts = s.get("counts", {})
        total_oq += counts.get("open_question", 0)
        total_cr += counts.get("complex_reflection", 0)
        total_counselor_words += s.get("counselor_words", 0)
        total_patient_words += s.get("patient_words", 0)
        total_turns += s.get("total_turns_scored", 0)

    total_words = total_counselor_words + total_patient_words

    def safe_ratio(num, denom):
        return round(num / denom, 4) if denom > 0 else None

    def safe_pct(num, denom):
        return round(num / denom * 100, 2) if denom > 0 else None

    return {
        "R_Q": safe_ratio(total_n_reflect, total_n_question),
        "pct_OQ": safe_pct(total_oq, total_n_question),
        "pct_CR": safe_pct(total_cr, total_n_reflect),
        "pct_MIC": safe_pct(total_n_mico, total_n_mico + total_n_miin),
        "pct_TTT": safe_pct(total_counselor_words, total_words),
        "n_samples": len(per_sample_scores),
        "total_turns_scored": total_turns,
        "total_n_reflect": total_n_reflect,
        "total_n_question": total_n_question,
        "total_n_mico": total_n_mico,
        "total_n_miin": total_n_miin,
        "total_counselor_words": total_counselor_words,
        "total_patient_words": total_patient_words,
    }
