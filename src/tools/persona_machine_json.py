"""
Unified parser and state-merge entry point for the Persona two-step LLM outputs
(short-term q + long-term p).

Lives at `src/tools/persona_machine_json.py`; depends on
`Module.prompts.extract_machine_json_line`, so the entry script must have
`FEP-Agent` on `sys.path` (the standard `import path_setup` convention).

Aligned with the MACHINE_JSON contract defined in `Module/prompts.py`:
- Short-term: **q_t** = `q_abstract[]` + `regularity_observation` + `q_delta`
  (containing an `add` object array and `cue_keyword`); compatible with the
  legacy `observation` key and string-typed `add`.
- Long-term: the **p(o|s)** delta is keyed on **latent_state -> reply_shape**;
  the **p(s'|s,a)** delta is keyed on **(s_from, action_a, s_to)**; compatible
  with the legacy `content` / `anchor` keys.

End-to-end usage:
1. `parse_short_term_llm_output` / `parse_long_term_llm_output` extract the
   structured result from the raw LLM text.
2. `apply_two_step_machine_json_to_personas` rewrites the **current**
   `_q_st_text`, `_observation_model_block`, `_transition_model_block` in
   place according to the MACHINE_JSON `q_delta` / `observation_model_delta`
   / `transition_model_delta` (drop -> modify -> add), then truncates to the
   length cap. It **no longer** appends the full CoT / table / bridge text to
   those three blocks.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from Module.prompts import extract_machine_json_line
from tools.therapist_action import normalize_therapist_action

PATIENT_CLIENT_TALK_TYPES_DEFAULT: Tuple[str, ...] = ("neutral", "change", "sustain")
CAMI_STAGE_TYPES: Tuple[str, ...] = ("precontemplation", "contemplation", "preparation")


def clamp_conf(x: Any, default: float = 0.5) -> float:
    try:
        v = float(x)
        return max(0.0, min(1.0, v))
    except (TypeError, ValueError):
        return default


def append_capped(prev: str, chunk: str, max_chars: int) -> str:
    """Append `chunk` after `prev` (separated by ---); when oversized, keep the tail."""
    chunk = (chunk or "").strip()
    if not chunk:
        p = (prev or "").strip()
        return p[:max_chars] if max_chars > 0 else p
    p = (prev or "").strip()
    merged = f"{p}\n---\n{chunk}" if p else chunk
    if max_chars <= 0 or len(merged) <= max_chars:
        return merged
    return merged[-max_chars:]


def strip_machine_json_prefix(text: str) -> str:
    """Strip the MACHINE_JSON marker line and its JSON, keeping the trailing CoT text."""
    if not text:
        return ""
    s = text.strip()
    m = re.search(r"MACHINE_JSON:\s*", s, re.IGNORECASE)
    if not m:
        return s
    rest = s[m.end() :].lstrip()
    try:
        _, end = json.JSONDecoder().raw_decode(rest)
        rest = rest[end:].lstrip()
    except json.JSONDecodeError:
        line_end = rest.find("\n")
        if line_end >= 0:
            rest = rest[line_end + 1 :].lstrip()
    return rest.strip()


def extract_q_manifold_section(text: str) -> str:
    """Extract the \"Final Posterior State Manifold q(s_t)\" panoramic description from the short-term CoT output."""
    if not text:
        return ""
    m = re.search(
        r"###\s*3\.\s*Final Posterior State Manifold[^\n]*\n(.*?)(?=\n###\s*4\.|\nMACHINE_JSON:|\Z)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return ""
    block = m.group(1).strip()
    m2 = re.search(
        r"\*\*Panoramic State Description\*\*:\s*(.*)",
        block,
        re.DOTALL | re.IGNORECASE,
    )
    if m2:
        return m2.group(1).strip()
    return block


def extract_long_term_markdown_section(raw: str, section_num: int) -> str:
    """Extract the body of the `### n.` section (excluding the heading line) from the long-term LLM reply."""
    if not raw:
        return ""
    m = re.search(
        rf"###\s*{section_num}\.\s*[^\n]*\n(.*?)(?=\n###\s*\d+\.|\Z)",
        raw,
        re.DOTALL | re.IGNORECASE,
    )
    return m.group(1).strip() if m else ""


def _normalize_stage_label(value: Any, patient_talk_types: Tuple[str, ...]) -> str:
    s = str(value or "").strip().lower()
    return s if s in patient_talk_types else ""


def _semantic_stage_fallback_from_patient_utt(
    patient_utt: str,
    patient_talk_types: Tuple[str, ...],
) -> str:
    """
    A lightweight CAMI semantic fallback parsed directly from the patient utterance when the short-term MACHINE_JSON fails.
    The goal is not to fully replace the LLM parser, but to avoid collapsing back to contemplation on clear planning / commitment turns.
    """
    if tuple(patient_talk_types) != CAMI_STAGE_TYPES:
        return ""
    text = str(patient_utt or "").strip().lower()
    if not text:
        return ""

    prep_patterns = (
        "i need to",
        "need to go",
        "need to talk to",
        "start taking steps",
        "take some steps",
        "first step",
        "next step",
        "i'm going to",
        "i am going to",
        "i'll",
        "i will",
        "this week",
        "next month",
        "plan to",
        "going to call",
        "talk to my boss",
    )
    if any(p in text for p in prep_patterns):
        return "preparation"

    contemplation_patterns = (
        "i know",
        "i guess",
        "probably",
        "might",
        "worried",
        "concerned",
        "i should",
        "i want to",
        "i'd like to",
        "i would like to",
        "maybe",
        "thinking about",
    )
    if any(p in text for p in contemplation_patterns):
        return "contemplation"

    precontemplation_patterns = (
        "don't think",
        "do not think",
        "not a problem",
        "don't know",
        "do not know",
        "won't work",
        "not going to work",
    )
    if any(p in text for p in precontemplation_patterns):
        return "precontemplation"

    return ""


def _recover_partial_short_term_machine_json(
    raw: str,
    patient_utt: str,
    fallback_client_talk_type: str,
    patient_talk_types: Tuple[str, ...],
    logger: Any = None,
) -> Dict[str, Any]:
    text = raw or ""
    recovered: Dict[str, Any] = {}

    ct = ""
    m = re.search(r'"client_talk_type"\s*:\s*"([^"\n]+)"', text, re.IGNORECASE)
    if m:
        ct = _normalize_stage_label(m.group(1), patient_talk_types)
    if not ct:
        m = re.search(r"client_talk_type\s*[:=]\s*([a-zA-Z_]+)", text, re.IGNORECASE)
        if m:
            ct = _normalize_stage_label(m.group(1), patient_talk_types)
    if not ct:
        ct = _semantic_stage_fallback_from_patient_utt(patient_utt, patient_talk_types)
    if not ct:
        ct = _normalize_stage_label(fallback_client_talk_type, patient_talk_types)
    if ct:
        recovered["client_talk_type"] = ct

    m = re.search(r'"inference_entropy"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text, re.IGNORECASE)
    if m:
        recovered["inference_entropy"] = clamp_conf(m.group(1), 0.5)

    if recovered and logger:
        logger.info(
            "[Persona] short_term_q recovered partial JSON fields=%s",
            sorted(recovered.keys()),
        )
    return recovered


def normalize_q_delta(raw: Any) -> Dict[str, Any]:
    """Parse `q_delta` from the short-term MACHINE_JSON, keeping the minimal structure: cue / state / note."""
    out: Dict[str, Any] = {"add_items": [], "modify": [], "drop": []}
    if not isinstance(raw, dict):
        return out
    ad = raw.get("add")
    if isinstance(ad, str) and ad.strip():
        out["add_items"].append({"cue": ad.strip()[:160], "state": "neutral", "note": ""})
    elif isinstance(ad, list):
        for it in ad:
            if isinstance(it, dict):
                cue = str(it.get("cue", "") or "").strip()
                st = str(it.get("state", "") or "").strip().lower()
                ts = str(it.get("transition_semantic", "") or "").strip()
                note = str(it.get("note", it.get("behavior_pattern", "")) or "").strip()
                if not cue and ts:
                    cue = ts
                su = it.get("state_update")
                if isinstance(su, dict):
                    s2 = str(su.get("state", "") or "").strip().lower()
                    if s2:
                        st = s2
                if cue:
                    out["add_items"].append(
                        {
                            "cue": cue,
                            "state": st or "neutral",
                            "note": note[:400],
                        }
                    )
            elif isinstance(it, str) and it.strip():
                out["add_items"].append({"cue": it.strip()[:160], "state": "neutral", "note": ""})
    for key in ("modify", "drop"):
        lst = raw.get(key) or []
        if not isinstance(lst, list):
            lst = []
        clean: List[Dict[str, str]] = []
        for it in lst:
            if isinstance(it, dict):
                qk = str(
                    it.get("cue_keyword", it.get("q_keyword", it.get("keyword", ""))) or ""
                ).strip()
                ns = str(it.get("new_state", "") or "").strip().lower()
                su = it.get("state_update")
                if isinstance(su, dict) and not ns:
                    ns = str(su.get("state", "") or "").strip().lower()
                note = str(it.get("note", it.get("behavior_pattern", "")) or "").strip()
                if qk or note or ns:
                    clean.append({"cue_keyword": qk, "new_state": ns, "note": note[:400]})
            elif isinstance(it, str) and it.strip():
                clean.append({"cue_keyword": it.strip(), "new_state": "", "note": ""})
        out[key] = clean
    return out


def _row_cue_match(entry_cue: str, kw: str) -> bool:
    if not kw:
        return False
    k, e = kw.strip(), (entry_cue or "").strip()
    return bool(e and (k == e or k in e or e in k))


def parse_q_abstract_snapshot(raw: Any, patient_talk_types: Tuple[str, ...]) -> List[Dict[str, str]]:
    """Parse a per-turn cue->state snapshot from MACHINE_JSON.q_abstract."""
    out: List[Dict[str, str]] = []
    if not isinstance(raw, list):
        return out
    for it in raw:
        if not isinstance(it, dict):
            continue
        cue = str(it.get("cue", it.get("keyword", "")) or "").strip()
        if not cue:
            continue
        st = str(it.get("state", it.get("client_talk_type", "")) or "").strip().lower()
        if st not in patient_talk_types:
            st = "neutral"
        out.append({"cue": cue, "state": st, "note": ""})
    return out


def merge_q_abstract_delta(
    rows: List[Dict[str, str]],
    q_delta: Dict[str, Any],
    patient_talk_types: Tuple[str, ...],
) -> List[Dict[str, str]]:
    """Apply `q_delta` (drop / modify / add_items) to the q_t row table."""
    cur: List[Dict[str, str]] = [dict(r) for r in rows]

    def idxs_for(kw: str) -> List[int]:
        return [i for i, r in enumerate(cur) if _row_cue_match(r.get("cue", ""), kw)]

    for item in q_delta.get("drop") or []:
        kw = str(item.get("cue_keyword", item.get("q_keyword", "")) or "").strip()
        if not kw:
            continue
        for i in reversed(idxs_for(kw)):
            cur.pop(i)

    for item in q_delta.get("modify") or []:
        kw = str(item.get("cue_keyword", "") or "").strip()
        new_st = str(item.get("new_state", "") or "").strip().lower()
        note = str(item.get("note", "") or "").strip()
        if not kw:
            continue
        for i in idxs_for(kw):
            if new_st in patient_talk_types:
                cur[i]["state"] = new_st
            if note:
                cur[i]["note"] = note[:400]

    for item in q_delta.get("add_items") or []:
        if not isinstance(item, dict):
            continue
        cue = str(item.get("cue", "") or "").strip()
        if not cue:
            continue
        st = str(item.get("state", "") or "").strip().lower()
        if st not in patient_talk_types:
            st = "neutral"
        note = str(item.get("note", "") or "").strip()
        cur = [r for r in cur if (r.get("cue") or "").strip() != cue]
        row: Dict[str, str] = {"cue": cue, "state": st, "note": note[:400]}
        cur.append(row)
    return cur


def upsert_q_abstract_snapshot(rows: List[Dict[str, str]], snapshot: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Overwrite rows with the same cue using this turn's q_abstract snapshot (snapshot wins)."""
    cur: List[Dict[str, str]] = [dict(r) for r in rows]
    for r in snapshot:
        c = (r.get("cue") or "").strip()
        if not c:
            continue
        cur = [x for x in cur if (x.get("cue") or "").strip() != c]
        cur.append(dict(r))
    return cur


def render_q_abstract_rows(rows: List[Dict[str, str]]) -> str:
    """Render the q_t row table as a structured JSON text."""
    clean = []
    for r in rows:
        c = (r.get("cue") or "").strip()
        if not c:
            continue
        clean.append(
            {
                "cue": c,
                "state": (r.get("state") or "neutral").strip(),
                "note": (r.get("note") or "").strip(),
            }
        )
    return json.dumps(clean, ensure_ascii=False, indent=2) if clean else ""


def cap_q_abstract_rows(rows: List[Dict[str, str]], max_rows: int) -> List[Dict[str, str]]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows
    return rows[-max_rows:]


def _coerce_q_rows(base: Any) -> List[Dict[str, str]]:
    if isinstance(base, list):
        out: List[Dict[str, str]] = []
        for it in base:
            if not isinstance(it, dict):
                continue
            cue = str(it.get("cue", "") or "").strip()
            if not cue:
                continue
            out.append(
                {
                    "cue": cue,
                    "state": str(it.get("state", "neutral") or "neutral").strip().lower(),
                    "note": str(it.get("note", "") or "").strip(),
                }
            )
        return out
    if isinstance(base, str) and base.strip():
        try:
            parsed = json.loads(base)
        except json.JSONDecodeError:
            return []
        return _coerce_q_rows(parsed)
    return []


def render_observation_rules(rules: List[Dict[str, str]]) -> str:
    clean = []
    for r in rules:
        rid = str(r.get("rule_id", "") or "").strip()
        if not rid:
            continue
        clean.append(
            {
                "rule_id": rid,
                "latent_state": str(r.get("latent_state", "") or "").strip(),
                "reply_shape": str(r.get("reply_shape", "") or "").strip(),
            }
        )
    return json.dumps(clean, ensure_ascii=False, indent=2) if clean else ""


def render_transition_rules(rules: List[Dict[str, str]]) -> str:
    clean = []
    for r in rules:
        rid = str(r.get("rule_id", "") or "").strip()
        if not rid:
            continue
        clean.append(
            {
                "rule_id": rid,
                "s_from": str(r.get("s_from", "") or "").strip(),
                "action_a": str(r.get("action_a", "") or "").strip(),
                "s_to": str(r.get("s_to", "") or "").strip(),
            }
        )
    return json.dumps(clean, ensure_ascii=False, indent=2) if clean else ""


def _cap_structured_items(
    items: List[Dict[str, str]],
    *,
    renderer,
    max_chars: int,
) -> List[Dict[str, str]]:
    cur = [dict(x) for x in items]
    if max_chars <= 0:
        return cur
    while cur and len(renderer(cur)) > max_chars:
        cur.pop(0)
    return cur


def _coerce_observation_rules(base: Any) -> List[Dict[str, str]]:
    if isinstance(base, list):
        out: List[Dict[str, str]] = []
        for it in base:
            if not isinstance(it, dict):
                continue
            rid = str(it.get("rule_id", "") or "").strip()
            if not rid:
                continue
            out.append(
                {
                    "rule_id": rid,
                    "latent_state": str(it.get("latent_state", "") or "").strip(),
                    "reply_shape": str(it.get("reply_shape", "") or "").strip(),
                }
            )
        return out
    if isinstance(base, str) and base.strip():
        try:
            parsed = json.loads(base)
        except json.JSONDecodeError:
            return []
        return _coerce_observation_rules(parsed)
    return []


def _coerce_transition_rules(base: Any) -> List[Dict[str, str]]:
    if isinstance(base, list):
        out: List[Dict[str, str]] = []
        for it in base:
            if not isinstance(it, dict):
                continue
            rid = str(it.get("rule_id", "") or "").strip()
            if not rid:
                continue
            out.append(
                {
                    "rule_id": rid,
                    "s_from": str(it.get("s_from", "") or "").strip(),
                    "action_a": str(it.get("action_a", "") or "").strip(),
                    "s_to": str(it.get("s_to", "") or "").strip(),
                }
            )
        return out
    if isinstance(base, str) and base.strip():
        try:
            parsed = json.loads(base)
        except json.JSONDecodeError:
            return []
        return _coerce_transition_rules(parsed)
    return []


def build_short_term_observation_payload(
    mj: Dict[str, Any],
    obs_fallback: Dict[str, Any],
    q_manifold: str,
    patient_utt: str,
    entropy: float,
) -> Dict[str, Any]:
    """Prefer `regularity_observation`; otherwise fall back to the legacy `observation` block."""
    reg = mj.get("regularity_observation")
    if isinstance(reg, dict):
        pre = str(reg.get("pre_regularities", "") or "").strip()
        post = str(reg.get("post_regularities", "") or "").strip()
        ctr = str(reg.get("contrast_regularities", "") or "").strip()
        merged = {
            "state_pattern": " | ".join(x for x in (pre, post) if x),
            "textual_cues": post[:400] if post else "",
            "behavioral_cues": pre[:400] if pre else "",
            "prediction_error": ctr[:800] if ctr else "",
            "confidence": reg.get("confidence"),
        }
        return finalize_short_term_observation(merged, q_manifold, patient_utt, entropy)
    obs = dict(obs_fallback) if isinstance(obs_fallback, dict) else {}
    coerce_observation_subfields(obs)
    return finalize_short_term_observation(obs, q_manifold, patient_utt, entropy)


def normalize_observation_model_delta(raw: Any) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = {"add": [], "modify": [], "drop": []}
    if not isinstance(raw, dict):
        return out
    for it in raw.get("add") or []:
        if not isinstance(it, dict):
            continue
        rid = str(it.get("rule_index", it.get("rule_id", "")) or "").strip()
        if not rid:
            continue
        out["add"].append(
            {
                "rule_id": rid,
                "latent_state": str(it.get("latent_state", it.get("state", "")) or "").strip(),
                "reply_shape": str(it.get("reply_shape", it.get("response_pattern", "")) or "").strip(),
            }
        )
    for it in raw.get("modify") or []:
        if not isinstance(it, dict):
            continue
        target = str(
            it.get("target_rule_index", it.get("target_rule_id", it.get("rule_index", it.get("rule_id", ""))))
            or ""
        ).strip()
        rid = str(it.get("updated_rule_index", it.get("rule_index", it.get("rule_id", target))) or "").strip()
        if not target and not rid:
            continue
        out["modify"].append(
            {
                "target_rule_id": target or rid,
                "rule_id": rid or target,
                "latent_state": str(it.get("latent_state", it.get("state", "")) or "").strip(),
                "reply_shape": str(it.get("reply_shape", it.get("response_pattern", "")) or "").strip(),
            }
        )
    for it in raw.get("drop") or []:
        if not isinstance(it, dict):
            continue
        target = str(
            it.get("target_rule_index", it.get("target_rule_id", it.get("rule_index", it.get("rule_id", ""))))
            or ""
        ).strip()
        if target:
            out["drop"].append({"target_rule_id": target})
    return out


def normalize_transition_model_delta(raw: Any) -> Dict[str, List[Dict[str, str]]]:
    out: Dict[str, List[Dict[str, str]]] = {"add": [], "modify": [], "drop": []}
    if not isinstance(raw, dict):
        return out
    for it in raw.get("add") or []:
        if not isinstance(it, dict):
            continue
        rid = str(it.get("rule_index", it.get("rule_id", "")) or "").strip()
        if not rid:
            continue
        out["add"].append(
            {
                "rule_id": rid,
                "s_from": str(it.get("s_from", it.get("source_state", "")) or "").strip(),
                "action_a": str(it.get("action_a", it.get("action", "")) or "").strip(),
                "s_to": str(it.get("s_to", it.get("target_state", "")) or "").strip(),
            }
        )
    for it in raw.get("modify") or []:
        if not isinstance(it, dict):
            continue
        target = str(
            it.get("target_rule_index", it.get("target_rule_id", it.get("rule_index", it.get("rule_id", ""))))
            or ""
        ).strip()
        rid = str(it.get("updated_rule_index", it.get("rule_index", it.get("rule_id", target))) or "").strip()
        if not target and not rid:
            continue
        out["modify"].append(
            {
                "target_rule_id": target or rid,
                "rule_id": rid or target,
                "s_from": str(it.get("s_from", it.get("source_state", "")) or "").strip(),
                "action_a": str(it.get("action_a", it.get("action", "")) or "").strip(),
                "s_to": str(it.get("s_to", it.get("target_state", "")) or "").strip(),
            }
        )
    for it in raw.get("drop") or []:
        if not isinstance(it, dict):
            continue
        target = str(
            it.get("target_rule_index", it.get("target_rule_id", it.get("rule_index", it.get("rule_id", ""))))
            or ""
        ).strip()
        if target:
            out["drop"].append({"target_rule_id": target})
    return out


def _infer_action_granularity_from_vocab(vocab: Tuple[str, ...]) -> str:
    for v in vocab or ():
        if "/" in str(v):
            return "fine"
    return "coarse"


def _normalize_transition_delta_actions(
    delta_raw: Any,
    therapist_actions: Tuple[str, ...],
) -> Any:
    if not isinstance(delta_raw, dict):
        return delta_raw
    granularity = _infer_action_granularity_from_vocab(therapist_actions)
    out = dict(delta_raw)
    for key in ("add", "modify"):
        items = out.get(key)
        if not isinstance(items, list):
            continue
        norm_items: List[Any] = []
        for it in items:
            if not isinstance(it, dict):
                norm_items.append(it)
                continue
            row = dict(it)
            raw_action = row.get("action_a", row.get("action"))
            if raw_action is not None:
                row["action_a"] = normalize_therapist_action(
                    str(raw_action),
                    granularity=granularity,
                    vocab=therapist_actions,
                    fallback_other=True,
                ) or "other"
            norm_items.append(row)
        out[key] = norm_items
    return out


def normalize_lt_p_delta(raw: Any) -> Optional[Dict[str, List[Dict[str, str]]]]:
    """Compatible with legacy logging / debug output; returns a pure-field delta."""
    if not isinstance(raw, dict):
        return None
    if any(k in json.dumps(raw, ensure_ascii=False) for k in ("latent_state", "reply_shape")):
        out = normalize_observation_model_delta(raw)
    else:
        out = normalize_transition_model_delta(raw)
    return out if any(out.values()) else None


def append_lt_delta_to_increment(composed: str, delta_raw: Any, section_title: str) -> str:
    norm = normalize_lt_p_delta(delta_raw)
    if not norm:
        return (composed or "").strip()
    blob = json.dumps(norm, ensure_ascii=False)
    block = f"### {section_title}\n{blob}"
    base = (composed or "").strip()
    return f"{base}\n\n{block}".strip() if base else block


def compose_observation_model_increment(sec1: str, sec2: str, omt: str) -> str:
    parts: List[str] = []
    if sec1:
        parts.append("### 1. Free Energy Derivation & Update Decision\n" + sec1)
    if sec2:
        parts.append("### 2. Observation Model Incremental Update\n" + sec2)
    if omt:
        parts.append("### Observation model update rationale (MACHINE_JSON)\n" + omt)
    return "\n\n".join(parts).strip()


def compose_transition_model_increment(sec1: str, sec3: str, tmt: str) -> str:
    parts: List[str] = []
    if sec1:
        parts.append("### 1. Free Energy Derivation & Update Decision\n" + sec1)
    if sec3:
        parts.append("### 3. Transition Model Incremental Update\n" + sec3)
    if tmt:
        parts.append("### Transition model update rationale (MACHINE_JSON)\n" + tmt)
    return "\n\n".join(parts).strip()


def compose_q_st_long_term_bridge(sec1: str, sec2: str, sec3: str) -> str:
    parts: List[str] = []
    if sec1:
        parts.append("[Long-run generative model — free energy / update decision]\n" + sec1[:900])
    tail_bits: List[str] = []
    if sec2:
        tail_bits.append("[p(o|s) table excerpt]\n" + sec2[:400])
    if sec3:
        tail_bits.append("[p(s'|s,a) dynamics excerpt]\n" + sec3[:400])
    if tail_bits:
        parts.append("\n".join(tail_bits))
    return "\n\n".join(parts).strip()


def coerce_observation_subfields(obs: Dict[str, Any]) -> None:
    """Normalize the observation sub-fields to short strings (in place)."""
    for _k in ("state_pattern", "textual_cues", "behavioral_cues", "prediction_error"):
        _v = obs.get(_k)
        if isinstance(_v, (dict, list)):
            obs[_k] = json.dumps(_v, ensure_ascii=False)[:800]
        elif _v is not None and not isinstance(_v, str):
            obs[_k] = str(_v)


def cap_tail_text(text: str, max_chars: int) -> str:
    """When a single text block exceeds the cap, keep the tail (matches the cap of the slow-changing / rolling q blocks)."""
    t = (text or "").strip()
    if max_chars <= 0:
        return t
    if len(t) <= max_chars:
        return t
    return t[-max_chars:]


def apply_long_term_block_delta(base: Any, delta_raw: Any) -> Any:
    """
    Apply a structured update to the observation / transition rule tables; only pure fields are kept.
    """
    if not isinstance(delta_raw, dict):
        return base if isinstance(base, list) else []

    base_is_observation = isinstance(base, list) and any(
        isinstance(it, dict) and ("latent_state" in it or "reply_shape" in it) for it in base
    )
    if base_is_observation or any(k in delta_raw for k in ("latent_state", "reply_shape")) or any(
        isinstance(it, dict) and ("latent_state" in it or "reply_shape" in it)
        for key in ("add", "modify")
        for it in (delta_raw.get(key) or [])
    ):
        rules = _coerce_observation_rules(base)
        d = normalize_observation_model_delta(delta_raw)

        for item in d.get("drop") or []:
            target = str(item.get("target_rule_id", "") or "").strip()
            rules = [r for r in rules if (r.get("rule_id") or "").strip() != target]

        for item in d.get("modify") or []:
            target = str(item.get("target_rule_id", "") or "").strip()
            for r in rules:
                if (r.get("rule_id") or "").strip() == target:
                    r["rule_id"] = str(item.get("rule_id", r.get("rule_id", "")) or "").strip()
                    if str(item.get("latent_state", "") or "").strip():
                        r["latent_state"] = str(item.get("latent_state", "") or "").strip()
                    if str(item.get("reply_shape", "") or "").strip():
                        r["reply_shape"] = str(item.get("reply_shape", "") or "").strip()

        for item in d.get("add") or []:
            rid = str(item.get("rule_id", "") or "").strip()
            if not rid:
                continue
            rules = [r for r in rules if (r.get("rule_id") or "").strip() != rid]
            rules.append(dict(item))
        return rules

    rules = _coerce_transition_rules(base)
    d = normalize_transition_model_delta(delta_raw)

    for item in d.get("drop") or []:
        target = str(item.get("target_rule_id", "") or "").strip()
        rules = [r for r in rules if (r.get("rule_id") or "").strip() != target]

    for item in d.get("modify") or []:
        target = str(item.get("target_rule_id", "") or "").strip()
        for r in rules:
            if (r.get("rule_id") or "").strip() == target:
                r["rule_id"] = str(item.get("rule_id", r.get("rule_id", "")) or "").strip()
                if str(item.get("s_from", "") or "").strip():
                    r["s_from"] = str(item.get("s_from", "") or "").strip()
                if str(item.get("action_a", "") or "").strip():
                    r["action_a"] = str(item.get("action_a", "") or "").strip()
                if str(item.get("s_to", "") or "").strip():
                    r["s_to"] = str(item.get("s_to", "") or "").strip()

    for item in d.get("add") or []:
        rid = str(item.get("rule_id", "") or "").strip()
        if not rid:
            continue
        rules = [r for r in rules if (r.get("rule_id") or "").strip() != rid]
        rules.append(dict(item))
    return rules


def apply_q_roll_delta(
    base: Any,
    qd: Dict[str, Any],
    *,
    snapshot_rows: Optional[List[Dict[str, str]]] = None,
    patient_talk_types: Tuple[str, ...] = PATIENT_CLIENT_TALK_TYPES_DEFAULT,
) -> List[Dict[str, str]]:
    """
    Execute the short-term `q_delta` on the structured q row table; only cue / state / note are kept.
    """
    rows = _coerce_q_rows(base)
    if snapshot_rows:
        rows = upsert_q_abstract_snapshot(rows, snapshot_rows)
    rows = merge_q_abstract_delta(rows, qd, patient_talk_types)
    return rows


def estimate_q_turn_state_probs(
    client_talk_type: Optional[str],
    *,
    inference_entropy: Optional[float] = None,
    observation_confidence: Optional[float] = None,
    patient_talk_types: Tuple[str, ...] = PATIENT_CLIENT_TALK_TYPES_DEFAULT,
) -> Dict[str, float]:
    """
    Compress the current-turn estimate of the latent state (client_talk_type, 3-class) into a distribution q(s_t) on the simplex.

    When the model does not output explicit probabilities, synthesize a distribution from the **point estimate label** + **inference_entropy** (higher -> flatter) +
    **observation.confidence** (higher -> sharper), for use by the Planner and the logger.

    CAMI stage calibration: LLM systematically over-predicts precontemplation
    (confusion matrix shows 43% of contemplation turns mis-labeled as precontemplation).
    When ct=precontemplation AND entropy is not very low, shift mass toward contemplation.
    """
    labels = tuple(patient_talk_types)
    if not labels:
        return {}
    ct = str(client_talk_type or "").strip().lower()
    if ct not in labels:
        # For CAMI stages, default to contemplation (midpoint) instead of labels[0]
        # to avoid systematic precontemplation bias
        if "contemplation" in labels:
            ct = "contemplation"
        elif "neutral" in labels:
            ct = "neutral"
        else:
            ct = labels[0]

    H = clamp_conf(inference_entropy, 0.5)
    conf_obs = clamp_conf(observation_confidence, 0.5)
    peak = (conf_obs + (1.0 - H)) / 2.0
    n = len(labels)
    peak = max(1.0 / n, min(0.97, peak))

    # CAMI stage calibration: mild softening of precontemplation predictions
    # Only apply when in CAMI stage space and prediction is precontemplation
    # with genuinely high entropy (uncertain prediction)
    _CAMI_STAGES = ("precontemplation", "contemplation", "preparation")
    if labels == _CAMI_STAGES and ct == "precontemplation" and H > 0.6 and peak < 0.75:
        # Small shift from precontemplation to contemplation when LLM is uncertain
        shift = 0.08 * (H - 0.6) / 0.4  # max shift ~0.08 when H=1.0
        probs = {}
        for lab in labels:
            if lab == ct:
                probs[lab] = peak - shift
            elif lab == "contemplation":
                probs[lab] = (1.0 - peak) / max(1, n - 1) + shift
            else:
                probs[lab] = (1.0 - peak) / max(1, n - 1)
        return probs

    rem = (1.0 - peak) / max(1, n - 1)
    return {lab: (peak if lab == ct else rem) for lab in labels}


def estimate_q_turn_state_probs_from_payload(
    data: Dict[str, Any],
    patient_talk_types: Tuple[str, ...] = PATIENT_CLIENT_TALK_TYPES_DEFAULT,
) -> Dict[str, float]:
    """Estimate q(s_t) from the payload shared by traits_update / apply (must include latent_mi and observation; inference_entropy is optional)."""
    lm = data.get("latent_mi") or {}
    ct = lm.get("client_talk_type") if isinstance(lm, dict) else None
    obs = data.get("observation") or {}
    oc = obs.get("confidence") if isinstance(obs, dict) else None
    H = data.get("inference_entropy")
    return estimate_q_turn_state_probs(
        ct,
        inference_entropy=H,
        observation_confidence=oc,
        patient_talk_types=patient_talk_types,
    )


def finalize_short_term_observation(
    obs: Dict[str, Any],
    q_manifold: str,
    patient_utt: str,
    entropy: float,
) -> Dict[str, Any]:
    """Fill empty fields and write `confidence` (returns a new dict, independent of the parser-side reference)."""
    out = dict(obs)
    if not str(out.get("state_pattern", "")).strip():
        out["state_pattern"] = (q_manifold[:220] + "…") if len(q_manifold) > 220 else q_manifold
    if not str(out.get("textual_cues", "")).strip():
        out["textual_cues"] = (patient_utt or "")[:280]
    if not str(out.get("behavioral_cues", "")).strip():
        out["behavioral_cues"] = f"inference_entropy={entropy:.2f}"
    if not str(out.get("prediction_error", "")).strip():
        out["prediction_error"] = ""
    out["confidence"] = clamp_conf(out.get("confidence"), entropy)
    return out


@dataclass
class ShortTermMachineParsed:
    """Short-term filtered MACHINE_JSON; the rolling-q state update uses `q_delta_normalized` as the source of truth."""

    machine_json: Dict[str, Any]
    client_talk_type: str
    inference_entropy: float
    observation: Dict[str, Any]
    q_delta_normalized: Dict[str, Any]
    q_manifold: str
    turn_body_for_q_roll: str  # By-product of the model output (e.g. for logging); not written back into `_q_st_text`.


@dataclass
class LongTermMachineParsed:
    """Long-term MACHINE_JSON; the slow-changing-block update uses the delta inside `machine_json` as the source of truth.

    `obs_increment` / `trans_increment` / `lt_bridge` are by-products of the CoT / table concatenation (debug only) and are not written into Personas.
    """

    machine_json: Dict[str, Any]
    therapist_action: str
    transition_inner: Dict[str, Any]
    observation_rationale: str
    transition_rationale: str
    obs_increment: str
    trans_increment: str
    lt_sec1: str
    lt_sec2: str
    lt_sec3: str
    lt_bridge: str


def parse_short_term_llm_output(
    raw: str,
    *,
    patient_utt: str,
    prior_q_full: str,
    fallback_client_talk_type: str,
    patient_talk_types: Tuple[str, ...] = PATIENT_CLIENT_TALK_TYPES_DEFAULT,
    logger: Any = None,
) -> ShortTermMachineParsed:
    mj = extract_machine_json_line(raw or "") or {}
    if not mj:
        mj = _recover_partial_short_term_machine_json(
            raw or "",
            patient_utt=patient_utt,
            fallback_client_talk_type=fallback_client_talk_type,
            patient_talk_types=patient_talk_types,
            logger=logger,
        )
    if not mj and logger:
        logger.warning(
            "[Persona] short_term_q: no MACHINE_JSON parsed; will fall back on client_talk_type / manifold heuristics."
        )

    q_manifold = extract_q_manifold_section(raw or "") or str(mj.get("q_st_manifold") or "").strip()
    if not q_manifold:
        sketch = f"(latent sketch from utterance) {(patient_utt or '')[:400]}"
        anchor = (prior_q_full or "")[-800:].strip() if prior_q_full else ""
        q_manifold = f"{anchor}\n{sketch}".strip() if anchor else sketch

    ct = str(mj.get("client_talk_type", "")).strip().lower()
    if ct not in patient_talk_types:
        semantic_ct = _semantic_stage_fallback_from_patient_utt(patient_utt, patient_talk_types)
        if semantic_ct:
            ct = semantic_ct
        else:
            ct = str(fallback_client_talk_type or "neutral").strip().lower()
            if ct not in patient_talk_types:
                ct = "neutral"
        if logger:
            logger.info(
                "[Persona] short_term_q client_talk_type fallback -> %s (invalid or missing in JSON)",
                ct,
            )

    entropy = clamp_conf(mj.get("inference_entropy"), 0.5)
    obs = mj.get("observation")
    if not isinstance(obs, dict):
        obs = {}
    else:
        obs = dict(obs)
    coerce_observation_subfields(obs)

    q_delta_norm = normalize_q_delta(mj.get("q_delta"))

    turn_body = strip_machine_json_prefix(raw or "") or (raw or "").strip()
    if q_manifold and q_manifold not in turn_body:
        turn_body = f"{turn_body}\n{q_manifold}".strip()

    return ShortTermMachineParsed(
        machine_json=mj,
        client_talk_type=ct,
        inference_entropy=entropy,
        observation=obs,
        q_delta_normalized=q_delta_norm,
        q_manifold=q_manifold,
        turn_body_for_q_roll=turn_body,
    )


def resolve_therapist_action_from_machine(
    mj: Dict[str, Any],
    vocab: Tuple[str, ...],
    override: Optional[str],
    logger: Any,
) -> str:
    """Prefer the caller-supplied override and normalize against the vocabulary; without an override, normalize the model label."""
    granularity = _infer_action_granularity_from_vocab(vocab)
    if override is not None:
        o = normalize_therapist_action(
            str(override),
            granularity=granularity,
            vocab=vocab,
            fallback_other=True,
        )
        if o:
            if logger:
                logger.debug("[Persona] long_term_p therapist_action from override=%s", o)
            return o
    ta = normalize_therapist_action(
        str(mj.get("therapist_action", "other")),
        granularity=granularity,
        vocab=vocab,
        fallback_other=True,
    ) or "other"
    if ta not in vocab:
        ta = "other"
        if logger and mj:
            logger.debug("[Persona] long_term_p therapist_action coerced -> other (not in vocabulary)")
    return ta


def _recover_partial_long_term_machine_json(
    raw: str,
    therapist_actions: Tuple[str, ...],
    action_override: Optional[str] = None,
    logger: Any = None,
) -> Dict[str, Any]:
    """
    Best-effort recovery for truncated / slightly malformed long-term fenced JSON.
    We only recover fields that are cheap and robust to match from raw text.
    """
    text = raw or ""
    recovered: Dict[str, Any] = {}

    if action_override is not None:
        act = str(action_override).strip().lower()
        if act:
            recovered["therapist_action"] = act

    ta = re.search(r'"therapist_action"\s*:\s*"([^"\n]+)"', text, re.IGNORECASE)
    if ta:
        act = ta.group(1).strip().lower()
        if act in therapist_actions:
            recovered["therapist_action"] = act

    conf = re.search(
        r'"transition"\s*:\s*\{.*?"confidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
        text,
        re.IGNORECASE | re.DOTALL,
    )
    rationale = re.search(
        r'"therapist_action_rationale"\s*:\s*"((?:\\.|[^"\\])*)"',
        text,
        re.IGNORECASE | re.DOTALL,
    )
    transition: Dict[str, Any] = {}
    if rationale:
        transition["therapist_action_rationale"] = bytes(
            rationale.group(1), "utf-8"
        ).decode("unicode_escape")
    if conf:
        transition["confidence"] = clamp_conf(conf.group(1), 0.5)
    if transition:
        recovered["transition"] = transition

    if recovered and logger:
        logger.info(
            "[Persona] long_term_p recovered partial JSON fields=%s",
            sorted(recovered.keys()),
        )
    return recovered


def parse_long_term_llm_output(
    raw: str,
    *,
    therapist_actions: Tuple[str, ...],
    action_override: Optional[str],
    logger: Any = None,
) -> LongTermMachineParsed:
    mj = extract_machine_json_line(raw or "") or {}
    if not mj:
        mj = _recover_partial_long_term_machine_json(
            raw or "",
            therapist_actions,
            action_override=action_override,
            logger=logger,
        )
    if isinstance(mj, dict) and mj:
        mj = dict(mj)
        if "transition_model_delta" in mj:
            mj["transition_model_delta"] = _normalize_transition_delta_actions(
                mj.get("transition_model_delta"),
                therapist_actions,
            )
    if not mj and logger:
        logger.warning(
            "[Persona] long_term_p: no MACHINE_JSON parsed; therapist_action defaults to other."
        )
    if logger and mj:
        logger.debug(
            "[Persona] long_term_p MACHINE_JSON therapist_action=%s transition_conf=%s "
            "obs_model_chars=%d trans_model_chars=%d",
            mj.get("therapist_action", "(missing)"),
            (mj.get("transition") or {}).get("confidence", "(n/a)")
            if isinstance(mj.get("transition"), dict)
            else "(n/a)",
            len(
                str(
                    mj.get("observation_model_update_rationale")
                    or mj.get("observation_model_text")
                    or ""
                )
            ),
            len(
                str(
                    mj.get("transition_model_update_rationale")
                    or mj.get("transition_model_text")
                    or ""
                )
            ),
        )
        od = normalize_lt_p_delta(mj.get("observation_model_delta"))
        td = normalize_lt_p_delta(mj.get("transition_model_delta"))
        if od or td:
            logger.debug(
                "[Persona] long_term_p MACHINE_JSON delta keys obs=%s trans=%s",
                list(od.keys()) if od else [],
                list(td.keys()) if td else [],
            )

    ta = resolve_therapist_action_from_machine(mj, therapist_actions, action_override, logger)
    tr = mj.get("transition") if isinstance(mj.get("transition"), dict) else {}
    omt = str(
        mj.get("observation_model_update_rationale") or mj.get("observation_model_text") or ""
    ).strip()
    tmt = str(
        mj.get("transition_model_update_rationale") or mj.get("transition_model_text") or ""
    ).strip()

    lt_sec1 = extract_long_term_markdown_section(raw or "", 1)
    lt_sec2 = extract_long_term_markdown_section(raw or "", 2)
    lt_sec3 = extract_long_term_markdown_section(raw or "", 3)

    obs_increment = compose_observation_model_increment(lt_sec1, lt_sec2, omt)
    obs_increment = append_lt_delta_to_increment(
        obs_increment,
        mj.get("observation_model_delta"),
        "observation_model_delta (MACHINE_JSON)",
    )
    trans_increment = compose_transition_model_increment(lt_sec1, lt_sec3, tmt)
    trans_increment = append_lt_delta_to_increment(
        trans_increment,
        mj.get("transition_model_delta"),
        "transition_model_delta (MACHINE_JSON)",
    )
    lt_bridge = compose_q_st_long_term_bridge(lt_sec1, lt_sec2, lt_sec3)

    if logger:
        tr_delta = str(
            tr.get("therapist_action_rationale") or tr.get("expected_state_delta") or ""
        ).replace("\n", " ")
        logger.info(
            "[Persona] long_term_p resolved action=%s therapist_action_rationale_excerpt=%s",
            ta,
            (tr_delta[:400] + "…") if len(tr_delta) > 400 else tr_delta or "(empty)",
        )
        logger.debug("[Persona] long_term_p omt=%s", omt)
        logger.debug("[Persona] long_term_p tmt=%s", tmt)
        logger.debug("[Persona] long_term_p tr=%s", tr)

    return LongTermMachineParsed(
        machine_json=mj,
        therapist_action=ta,
        transition_inner=dict(tr) if isinstance(tr, dict) else {},
        observation_rationale=omt,
        transition_rationale=tmt,
        obs_increment=obs_increment,
        trans_increment=trans_increment,
        lt_sec1=lt_sec1,
        lt_sec2=lt_sec2,
        lt_sec3=lt_sec3,
        lt_bridge=lt_bridge,
    )


def apply_two_step_machine_json_to_personas(
    personas: Any,
    st: ShortTermMachineParsed,
    lt: LongTermMachineParsed,
    *,
    patient_utt: str,
    therapist_utt: str,
    prior_q_full: str,
    max_history: int,
    logger: Any = None,
) -> Dict[str, Any]:
    """
    Update the Personas internal state from the parsed short-/long-term MACHINE_JSON and return a dict consumable by `_apply_payload`.

    All **three state blocks** are structured lists:
    - `_q_st_text`: `[{cue, state, note}]`
    - `_observation_model_block`: `[{rule_id, latent_state, reply_shape}]`
    - `_transition_model_block`: `[{rule_id, s_from, action_a, s_to}]`

    Also updates `_history_buffer`, `_last_inference_entropy`, and the `_prev_*` fields.
    """
    st_mj = st.machine_json if isinstance(getattr(st, "machine_json", None), dict) else {}
    obs = build_short_term_observation_payload(
        st_mj,
        st.observation,
        st.q_manifold,
        patient_utt,
        st.inference_entropy,
    )
    ct = st.client_talk_type
    ta = lt.therapist_action
    tr = lt.transition_inner
    mj = lt.machine_json
    patient_talk_types = tuple(getattr(personas, "_patient_talk_types", PATIENT_CLIENT_TALK_TYPES_DEFAULT))

    qd = st.q_delta_normalized
    q_snapshot = parse_q_abstract_snapshot(st_mj.get("q_abstract"), patient_talk_types)
    personas._q_st_text = _cap_structured_items(
        apply_q_roll_delta(
            personas._q_st_text,
            qd,
            snapshot_rows=q_snapshot,
            patient_talk_types=patient_talk_types,
        ),
        renderer=render_q_abstract_rows,
        max_chars=personas._q_inference_cap,
    )

    personas._observation_model_block = _cap_structured_items(
        apply_long_term_block_delta(
            personas._observation_model_block,
            mj.get("observation_model_delta"),
        ),
        renderer=render_observation_rules,
        max_chars=personas._slow_obs_model_cap,
    )
    personas._transition_model_block = _cap_structured_items(
        apply_long_term_block_delta(
            personas._transition_model_block,
            mj.get("transition_model_delta"),
        ),
        renderer=render_transition_rules,
        max_chars=personas._slow_trans_model_cap,
    )

    personas._history_buffer.append(
        {
            "q_s_prior_excerpt": (prior_q_full or "")[:500],
            "a_prev_label": personas._prev_therapist_action,
            "a_prev_utt_excerpt": (personas._prev_therapist_utt or "")[:300],
            "q_s_post_excerpt": st.q_manifold[:600],
            "o_t_excerpt": (patient_utt or "")[:300],
            "therapist_utt_excerpt": (therapist_utt or "")[:300],
            "posterior_state": ct,
            "current_action_label": ta,
        }
    )
    while len(personas._history_buffer) > max_history:
        personas._history_buffer.pop(0)

    personas._last_inference_entropy = st.inference_entropy
    personas._prev_therapist_utt = therapist_utt
    personas._prev_therapist_action = ta

    if logger:
        logger.debug(
            "[Persona] long_term_p observation_model_block=%s",
            render_observation_rules(personas._observation_model_block),
        )
        logger.debug(
            "[Persona] long_term_p transition_model_block=%s",
            render_transition_rules(personas._transition_model_block),
        )
        logger.debug("[Persona] long_term_p prior_q_full=%s", prior_q_full)
        logger.debug("[Persona] long_term_p q_manifold=%s", st.q_manifold)
        logger.debug("[Persona] long_term_p entropy=%s", st.inference_entropy)
        logger.debug("[Persona] short_term_q self._q_st_text=%s", render_q_abstract_rows(personas._q_st_text))

    return {
        "observation": obs,
        "latent_mi": {"client_talk_type": ct},
        "inference_entropy": st.inference_entropy,
        "therapist_action": ta,
        "q_delta": st.q_delta_normalized,
        "transition": {
            "expected_state_delta": str(
                tr.get("therapist_action_rationale") or tr.get("expected_state_delta", "")
            ).strip()
            or f"Post-action expectation under a_t={ta} (see long-term transition block).",
            "confidence": clamp_conf(tr.get("confidence"), 0.5),
        },
    }


__all__ = [
    "PATIENT_CLIENT_TALK_TYPES_DEFAULT",
    "ShortTermMachineParsed",
    "LongTermMachineParsed",
    "append_capped",
    "apply_long_term_block_delta",
    "apply_q_roll_delta",
    "apply_two_step_machine_json_to_personas",
    "cap_tail_text",
    "clamp_conf",
    "estimate_q_turn_state_probs",
    "estimate_q_turn_state_probs_from_payload",
    "compose_observation_model_increment",
    "compose_q_st_long_term_bridge",
    "compose_transition_model_increment",
    "extract_long_term_markdown_section",
    "extract_q_manifold_section",
    "finalize_short_term_observation",
    "normalize_lt_p_delta",
    "normalize_q_delta",
    "parse_long_term_llm_output",
    "parse_short_term_llm_output",
    "resolve_therapist_action_from_machine",
    "strip_machine_json_prefix",
]
