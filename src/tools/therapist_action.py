from __future__ import annotations

from typing import Optional, Sequence, Set


_COARSE_CORE: Set[str] = {"question", "reflection", "therapist_input", "other"}
_FINE_CORE: Set[str] = {
    "question/open",
    "question/closed",
    "reflection/simple",
    "reflection/complex",
    "therapist_input/information",
    "therapist_input/advice",
    "therapist_input/negotiation",
    "therapist_input/options",
    "other",
}


_COARSE_ALIAS = {
    "input": "therapist_input",
    "inform": "therapist_input",
    "information": "therapist_input",
    "advice": "therapist_input",
    "negotiation": "therapist_input",
    "options": "therapist_input",
    "plan": "therapist_input",
    "therapistinput": "therapist_input",
    "therapist-input": "therapist_input",
    "acknowledge": "other",
    "engage": "other",
    "downplay": "other",
    "hesitate": "other",
    "accept": "other",
    "doubt": "other",
    "blame": "other",
    "deny": "other",
    "reject": "other",
}

_FINE_ALIAS = {
    "question": "question/open",
    "reflection": "reflection/simple",
    "therapist_input": "therapist_input/information",
    "input": "therapist_input/information",
    "inform": "therapist_input/information",
    "information": "therapist_input/information",
    "advice": "therapist_input/advice",
    "negotiation": "therapist_input/negotiation",
    "options": "therapist_input/options",
    "plan": "therapist_input/options",
    "acknowledge": "other",
    "engage": "other",
    "downplay": "other",
    "hesitate": "other",
    "accept": "other",
    "doubt": "other",
    "blame": "other",
    "deny": "other",
    "reject": "other",
}


def canonicalize_action_token(action: Optional[str]) -> str:
    if action is None:
        return ""
    return str(action).strip().lower().replace(" ", "_")


def normalize_therapist_action(
    action: Optional[str],
    *,
    granularity: str = "coarse",
    vocab: Optional[Sequence[str]] = None,
    fallback_other: bool = True,
) -> Optional[str]:
    tok = canonicalize_action_token(action)
    if not tok:
        return None

    g = (granularity or "coarse").strip().lower()
    if g == "fine":
        if tok in _FINE_ALIAS:
            tok = _FINE_ALIAS[tok]
        elif "/" in tok:
            head, tail = tok.split("/", 1)
            head = _COARSE_ALIAS.get(head, head)
            tok = f"{head}/{tail}"
        if tok not in _FINE_CORE:
            tok = "other" if fallback_other else ""
    else:
        if "/" in tok:
            tok = tok.split("/", 1)[0].strip()
        tok = _COARSE_ALIAS.get(tok, tok)
        if tok not in _COARSE_CORE:
            tok = "other" if fallback_other else ""

    if not tok:
        return None
    if not vocab:
        return tok

    vset = set(vocab)
    if tok in vset:
        return tok
    if fallback_other and "other" in vset:
        return "other"
    return None

