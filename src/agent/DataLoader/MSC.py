import os
import json
import re
import numpy as np
from typing import List, Optional, Tuple

import path_setup  # noqa: F401
from counselor import LocalModel
from tools.persona_machine_json import normalize_therapist_action

from nltk.util import ngrams
from nlgeval import calc_nlg_metrics
from Module.EventMemory import EventMemory
from Module.Personas import Personas, therapist_action_vocab


MI_TALK_TYPES = ("neutral", "change", "sustain")
CAMI_STAGE_TYPES = ("precontemplation", "contemplation", "preparation")

from Module.Generator import Generator
from Module.Planner import TransitionPlanner
from Module.prompts import STATE_ALIGNMENT_SYS, format_state_alignment_user
from Module.MITIScorer import MITIScorer, aggregate_miti_scores


NLG_METRIC_NAMES = (
    "B-1",
    "B-2",
    "B-3",
    "B-4",
    "R-L",
    "Dist-1",
    "Dist-2",
    "Dist-3",
)
CURRENT_STATE_METRIC_NAME = "Current-State-Acc"
NEXT_STATE_METRIC_NAME = "Next-State-Eval"
METRIC_NAMES = NLG_METRIC_NAMES + (CURRENT_STATE_METRIC_NAME, NEXT_STATE_METRIC_NAME)


def _parse_sample_ids_arg(raw):
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    ids = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        ids.append(int(token))
    return ids or None


def utterances_to_patient_doctor_pairs(utterances):
    """
    AnnoMI: speaker 'client' = patient (SPEAKER_1), next 'therapist' = doctor (SPEAKER_2).
    """
    pairs = []
    n = len(utterances)
    i = 0
    while i < n:
        if utterances[i].get("speaker") == "client":
            j = i + 1
            while j < n and utterances[j].get("speaker") != "therapist":
                j += 1
            if j < n:
                pair = {
                    "SPEAKER_1": utterances[i].get("utterance_text", ""),
                    "SPEAKER_2": utterances[j].get("utterance_text", ""),
                    "client_talk_type": utterances[i].get("client_talk_type"),
                }
                th = utterances[j]
                for key in (
                    "therapist_action_fine_qualified",
                    "therapist_action_coarse",
                    "main_therapist_behaviour",
                    "therapist_action",
                ):
                    v = th.get(key)
                    if v is not None and str(v).strip():
                        pair["therapist_action"] = str(v).strip().lower().replace(" ", "_")
                        break
                pairs.append(pair)
                i = j + 1
            else:
                i += 1
        else:
            i += 1
    return pairs


def _split_pairs_as_two_sessions(pairs, init_turns, warmup_ratio=None, min_eval_turns=1):
    n_pairs = len(pairs)
    if n_pairs <= 1:
        split = 0
    elif warmup_ratio is not None:
        ratio = max(0.0, min(1.0, float(warmup_ratio)))
        min_eval = max(1, int(min_eval_turns))
        max_warmup = max(0, n_pairs - min_eval)
        target = int(round(n_pairs * ratio))
        min_warmup = 1 if n_pairs > 1 else 0
        split = min(max(min_warmup, target), max_warmup)
        split = min(split, n_pairs - 1)
    else:
        split = min(max(0, int(init_turns)), max(0, n_pairs - 1))
    init_pairs = pairs[:split]
    eval_pairs = pairs[split:]
    if not eval_pairs:
        eval_pairs = pairs[-1:]
        init_pairs = pairs[:-1] if len(pairs) > 1 else []
    return init_pairs, eval_pairs


def load_annomi_jsonl_as_samples(
    path,
    init_turns,
    test_num,
    logger,
    warmup_ratio=None,
    min_eval_turns=1,
):
    """
    Each JSONL line -> one sample: [session0 warm-up dialog, session1 eval dialog].
    time_pass unused (kept as 0 for schema compatibility).

    When *warmup_ratio* is not None the split uses ratio-based warmup
    (same logic as CAMI) instead of the fixed *init_turns* count.
    """
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            utt = rec.get("utterances") or []
            pairs = utterances_to_patient_doctor_pairs(utt)
            if not pairs:
                logger.info(
                    f"Skip transcript_id={rec.get('transcript_id')}: no client→therapist pairs."
                )
                continue
            init_pairs, eval_pairs = _split_pairs_as_two_sessions(
                pairs,
                init_turns,
                warmup_ratio=warmup_ratio,
                min_eval_turns=min_eval_turns,
            )
            samples.append(
                [
                    {"dialog": init_pairs, "time_pass": 0},
                    {"dialog": eval_pairs, "time_pass": 0},
                ]
            )
    if test_num > 0:
        samples = samples[:test_num]
    logger.info(f"Loaded {len(samples)} AnnoMI transcripts from {path}.")
    return samples


def load_cami_profiles_jsonl_as_samples(
    path,
    init_turns,
    test_num,
    logger,
    warmup_ratio=None,
    min_eval_turns=1,
):
    """
    Each JSONL line -> one CAMI profile with fields like speakers/utterances/talk_types/actions.
    Converted into [session0 warm-up dialog, session1 eval dialog] with client→counselor pairs.

    Gold labels for CAMI evaluation can be:
    - talk_types: {neutral, change, sustain}
    - states: {precontemplation, contemplation, preparation}

    We store both into the dialog pairs and select the evaluation key later.
    """
    samples = []

    def _norm_talk_type(v):
        if v is None:
            return None
        s = str(v).strip().lower()
        if s in MI_TALK_TYPES:
            return s
        return None

    def _norm_stage(v):
        if v is None:
            return None
        s = str(v).strip().lower()
        if s in CAMI_STAGE_TYPES:
            return s
        return None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            speakers = rec.get("speakers") or []
            utterances = rec.get("utterances") or []
            talk_types = rec.get("talk_types") or []
            states = rec.get("states") or []
            actions = rec.get("actions") or []

            n = min(len(speakers), len(utterances), len(talk_types), len(states), len(actions))
            if n <= 1:
                continue
            speakers = speakers[:n]
            utterances = utterances[:n]
            talk_types = talk_types[:n]
            states = states[:n]
            actions = actions[:n]

            pairs = []
            i = 0
            while i < n:
                if speakers[i] == "client":
                    j = i + 1
                    while j < n and speakers[j] != "counselor":
                        j += 1
                    if j < n:
                        pair = {
                            "SPEAKER_1": utterances[i] or "",
                            "SPEAKER_2": utterances[j] or "",
                            "client_talk_type": _norm_talk_type(talk_types[i]),
                            "client_stage": _norm_stage(states[i]),
                        }
                        a = actions[j]
                        if a is not None and str(a).strip():
                            pair["therapist_action"] = (
                                str(a).strip().lower().replace(" ", "_")
                            )
                        pairs.append(pair)
                        i = j + 1
                    else:
                        i += 1
                else:
                    i += 1

            if not pairs:
                logger.info(f"Skip CAMI idx={rec.get('idx')}: no client→counselor pairs.")
                continue

            init_pairs, eval_pairs = _split_pairs_as_two_sessions(
                pairs,
                init_turns,
                warmup_ratio=warmup_ratio,
                min_eval_turns=min_eval_turns,
            )
            samples.append(
                [
                    {"dialog": init_pairs, "time_pass": 0},
                    {"dialog": eval_pairs, "time_pass": 0},
                ]
            )

    if test_num > 0:
        samples = samples[:test_num]
    logger.info(f"Loaded {len(samples)} CAMI profiles from {path}.")
    return samples


def format_turns_ago(delta_turns):
    if delta_turns <= 0:
        return "0 turns"
    return f"{int(delta_turns)} turns"


def resolve_persona_therapist_action_for_traits(dial, args, logger):
    """
    Merge per-turn annotations with the global `--persona_therapist_action` and validate against the `therapist_action_granularity` vocabulary at this layer.
    If the value is not in the vocabulary, log and return None (Personas then lets the model annotate, avoiding silent rewrites).
    """
    per_turn_action = dial.get("therapist_action")
    global_action = (getattr(args, "persona_therapist_action", "") or "").strip() or None
    raw = per_turn_action or global_action
    if raw is None:
        return None
    a = str(raw).strip()
    if not a:
        return None
    granularity = getattr(args, "therapist_action_granularity", "coarse")
    vocab = therapist_action_vocab(granularity)
    a_norm = normalize_therapist_action(
        a,
        granularity=granularity,
        vocab=vocab,
        fallback_other=True,
    )
    if a_norm is None:
        if logger:
            logger.warning(
                "[MSC] therapist_action %r not normalizable (granularity=%s); "
                "omitting persona override (LLM will label).",
                raw,
                granularity,
            )
        return None
    raw_norm = str(raw).strip().lower().replace(" ", "_")
    if logger and a_norm != raw_norm:
        logger.debug(
            "[MSC] therapist_action normalized: raw=%r -> normalized=%r (granularity=%s)",
            raw,
            a_norm,
            granularity,
        )
    return a_norm


def _extract_json_object(text):
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


def _normalize_mi_talk_type(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in MI_TALK_TYPES:
        return s
    return None


def _normalize_cami_stage(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in CAMI_STAGE_TYPES:
        return s
    return None


def _extract_mi_talk_type_from_text(text: str) -> Optional[str]:
    data = _extract_json_object(text)
    if isinstance(data, dict):
        direct = _normalize_mi_talk_type(data.get("client_talk_type"))
        if direct:
            return direct
        latent_next = data.get("latent_mi_next")
        if isinstance(latent_next, dict):
            nested = _normalize_mi_talk_type(latent_next.get("client_talk_type"))
            if nested:
                return nested
        latent = data.get("latent_mi")
        if isinstance(latent, dict):
            nested = _normalize_mi_talk_type(latent.get("client_talk_type"))
            if nested:
                return nested

    m = re.search(r"client_talk_type\s*[:=]\s*([a-zA-Z_]+)", text or "", re.IGNORECASE)
    if m:
        return _normalize_mi_talk_type(m.group(1))
    return None


def _extract_cami_stage_from_text(text: str) -> Optional[str]:
    data = _extract_json_object(text)
    if isinstance(data, dict):
        # Compatible with both the CAMI native field and the Personas unified field (client_talk_type stores the CAMI stage).
        direct = _normalize_cami_stage(data.get("client_stage"))
        if direct:
            return direct
        direct = _normalize_cami_stage(data.get("client_talk_type"))
        if direct:
            return direct
        latent_next = data.get("latent_mi_next")
        if isinstance(latent_next, dict):
            nested = _normalize_cami_stage(latent_next.get("client_stage"))
            if nested:
                return nested
            nested = _normalize_cami_stage(latent_next.get("client_talk_type"))
            if nested:
                return nested
        latent = data.get("latent_mi")
        if isinstance(latent, dict):
            nested = _normalize_cami_stage(latent.get("client_stage"))
            if nested:
                return nested
            nested = _normalize_cami_stage(latent.get("client_talk_type"))
            if nested:
                return nested

    # Fallback to regex: match both `client_stage` and lines like `client_talk_type: precontemplation`.
    for field in ("client_stage", "client_talk_type"):
        m = re.search(rf"{field}\s*[:=]\s*([a-zA-Z_]+)", text or "", re.IGNORECASE)
        if m:
            v = _normalize_cami_stage(m.group(1))
            if v:
                return v
    return None


def _safe_nanmean(vectors):
    if not vectors:
        return np.full(len(METRIC_NAMES), np.nan)
    arr = np.array(vectors, dtype=float)
    mean = np.full(arr.shape[1], np.nan)
    for idx in range(arr.shape[1]):
        col = arr[:, idx]
        valid = col[~np.isnan(col)]
        if valid.size > 0:
            mean[idx] = valid.mean()
    return mean


def _format_metric_triplets(score):
    parts = []
    for idx, name in enumerate(METRIC_NAMES):
        val = score[idx]
        if np.isnan(val):
            parts.append(f"{name}: N/A")
        else:
            parts.append(f"{name}: {val}")
    return ";  ".join(parts)


def _metric_dict(score):
    data = {}
    for idx, name in enumerate(METRIC_NAMES):
        val = float(score[idx]) if not np.isnan(score[idx]) else None
        data[name] = val
    return data


class MSC():
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.sampling_dataset = []
        self.original_sample_ids = None
        self._concise_static_logging = bool(getattr(args, "concise_static_logging", True))

        self._annomi_mode = args.dataset in ("annomi", "kmi") or getattr(args, "data_format", "msc") == "annomi_jsonl"
        self._cami_mode = args.dataset == "cami" or getattr(args, "data_format", "msc") == "cami_jsonl"
        self._jsonl_mi_mode = self._annomi_mode or self._cami_mode
        self.client = LocalModel(args)

        self.usr_name = args.usr_name
        self.agent_name = args.agent_name

        if self._annomi_mode:
            annomi_path = os.path.join(args.data_path, args.data_name)
            self._raw_data_path = annomi_path
            self.dataset = load_annomi_jsonl_as_samples(
                annomi_path,
                args.annomi_init_turns,
                args.test_num,
                logger,
                warmup_ratio=getattr(args, "cami_warmup_ratio", None),
                min_eval_turns=getattr(args, "cami_min_eval_turns", 3),
            )
        elif self._cami_mode:
            cami_path = os.path.join(args.data_path, args.data_name)
            self._raw_data_path = cami_path
            self.dataset = load_cami_profiles_jsonl_as_samples(
                cami_path,
                args.annomi_init_turns,
                args.test_num,
                logger,
                warmup_ratio=getattr(args, "cami_warmup_ratio", 0.5),
                min_eval_turns=getattr(args, "cami_min_eval_turns", 3),
            )
        else:
            self._raw_data_path = os.path.join(args.data_path, args.data_name)
            with open(os.path.join(args.data_path, args.data_name), "r", encoding="utf-8") as f:
                self.dataset = json.load(f)
                if args.test_num > 0:
                    self.dataset = self.dataset[: args.test_num]

        selected_ids = _parse_sample_ids_arg(getattr(args, "sample_ids", ""))
        if selected_ids is not None:
            max_idx = len(self.dataset) - 1
            bad_ids = [idx for idx in selected_ids if idx < 0 or idx > max_idx]
            if bad_ids:
                raise IndexError(
                    f"--sample_ids contains out-of-range ids {bad_ids}; valid range is [0, {max_idx}]"
                )
            self.dataset = [self.dataset[idx] for idx in selected_ids]
            self.original_sample_ids = list(selected_ids)
            self.logger.info("Selected original sample ids: %s", self.original_sample_ids)
        else:
            self.original_sample_ids = list(range(len(self.dataset)))

        self.logger.info(f"Total {len(self.dataset)} samples to be evaluated.")

        self.relevance_memory_number = args.relevance_memory_number
        self.context_memory_number = args.context_memory_number
        self.dist_thres = args.dist_thres
        self.memory_cache = args.memory_cache

        self.overall_retrieve_score = 0.0
        self.overall_retrieve_count = 0

        self._dialog_turn = 0
        self._state_label_space = getattr(args, "state_label_space", "mi_talk")
        self._state_key = "client_stage" if self._state_label_space == "cami_stage" else "client_talk_type"
        self._state_labels = CAMI_STAGE_TYPES if self._state_label_space == "cami_stage" else MI_TALK_TYPES
        # Confusion data for Macro-F1 computation
        self._confusion_pairs: List[Tuple[str, str]] = []  # (pred, gold)


        # MITI Behavior Scores tracking (enabled by --miti_behavior_scores)
        self._miti_enabled = bool(getattr(args, "miti_behavior_scores", False))
        self._miti_per_sample = []  # list of per-sample score dicts
        if self._miti_enabled:
            self._miti_scorer = MITIScorer(self.client, self.logger)
            self.logger.info("[MITI] Using the local model for behavior classification.")

    def _static_log_info(self, msg, *args):
        if self._concise_static_logging:
            self.logger.debug(msg, *args)
        else:
            self.logger.info(msg, *args)


    def _build_results_payload(self, all_samples_score, all_sample_avg, min_s, max_s):
        samples_payload = []
        for sample_id, (session_scores, sample_avg) in enumerate(zip(all_samples_score, all_sample_avg)):
            sample_item = {
                "sample_id": sample_id,
                "original_sample_id": (
                    self.original_sample_ids[sample_id]
                    if self.original_sample_ids is not None and sample_id < len(self.original_sample_ids)
                    else sample_id
                ),
                "sample_average": _metric_dict(sample_avg),
                "miti_behavior_scores": (
                    self._miti_per_sample[sample_id]
                    if self._miti_per_sample and sample_id < len(self._miti_per_sample)
                    else None
                ),
                "sessions": [],
            }
            for session_offset, session_score in enumerate(session_scores):
                session_number = min_s + session_offset + 1
                sample_item["sessions"].append(
                    {
                        "session_number": session_number,
                        "metrics": _metric_dict(session_score),
                    }
                )
            samples_payload.append(sample_item)

        session_payload = []
        for session_number in range(min_s + 1, max_s + 1):
            score_session = [
                sample[session_number - min_s - 1]
                for sample in all_samples_score
                if len(sample) > (session_number - min_s - 1)
            ]
            session_payload.append(
                {
                    "session_number": session_number,
                    "metrics": _metric_dict(_safe_nanmean(score_session)),
                }
            )

        overall_avg = _safe_nanmean(all_sample_avg)
        classification_metrics = self._compute_classification_metrics()
        miti_aggregate = aggregate_miti_scores(self._miti_per_sample) if self._miti_per_sample else None
        return {
            "classification_metrics": classification_metrics,
            "miti_behavior_scores": miti_aggregate,
            "run": {
                "dataset": self.args.dataset,
                "method": "puma",
                "data_format": getattr(self.args, "data_format", None),
                "test_num": self.args.test_num,
                "sample_ids": getattr(self.args, "sample_ids", ""),
                "build_times": self.args.build_times,
                "disable_planner": bool(getattr(self.args, "disable_planner", False)),
                "next_state_eval_mode": getattr(self.args, "next_state_eval_mode", "auto"),
                "log_file": getattr(self.args, "log_file", None),
                "results_file": getattr(self.args, "results_file", None),
                "metric_names": list(METRIC_NAMES),
            },
            "aggregate": {
                "n_samples": len(all_samples_score),
                "overall_average": _metric_dict(overall_avg),
                "by_session": session_payload,
            },
            "samples": samples_payload,
        }

    def _compute_classification_metrics(self):
        """Compute per-class precision/recall/F1 and Macro-F1 from accumulated confusion pairs."""
        if not self._confusion_pairs:
            return {}
        labels = list(self._state_labels)
        per_class = {}
        for label in labels:
            tp = sum(1 for p, g in self._confusion_pairs if p == label and g == label)
            fp = sum(1 for p, g in self._confusion_pairs if p == label and g != label)
            fn = sum(1 for p, g in self._confusion_pairs if p != label and g == label)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            support = tp + fn
            per_class[label] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        macro_f1 = sum(c["f1"] for c in per_class.values()) / len(per_class) if per_class else 0.0
        macro_recall = sum(c["recall"] for c in per_class.values()) / len(per_class) if per_class else 0.0
        return {
            "macro_f1": round(macro_f1, 4),
            "macro_recall": round(macro_recall, 4),
            "per_class": per_class,
            "total_turns": len(self._confusion_pairs),
        }

    def _save_results_json(self, payload):
        out_path = getattr(self.args, "results_file", None)
        if not out_path:
            return
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self.logger.info("Saved metrics JSON to %s", out_path)
        latest_out_path = getattr(self.args, "results_latest_file", None)
        if latest_out_path:
            os.makedirs(os.path.dirname(latest_out_path), exist_ok=True)
            with open(latest_out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self.logger.info("Saved latest metrics JSON to %s", latest_out_path)

    def _session_range(self):
        if self._jsonl_mi_mode:
            # Same layout as MSC: 0 = warmup, 1 = evaluation. range(min, max) is half-open, so max=2 actually runs session 1.
            return 1, 2
        return self.args.min_session, self.args.max_session

    def memory_bank_init(self, sample_id, args):
        """
        Initialize first session (warm-up turns for AnnoMI / session 0 for MSC).

        When observation_focused_state=True, also runs traits_update during warmup
        so that p(o|s) and p(s'|s,a) accumulate rules based on (classifier-corrected)
        state labels. This gives the FEP world model a good foundation before eval.
        """
        self.logger.info(f'Memory ID: {sample_id}')
        memory_bank = EventMemory(self.client, sample_id=sample_id, logger=self.logger, args=args, memory_cache=self.memory_cache)
        personas = Personas(self.client, logger=self.logger, args=args)
        response_generator = Generator(self.client, self.sampling_dataset, sample_id, logger=self.logger, args=args)
        planner = TransitionPlanner(self.client, self.logger, args)

        warmup_with_personas = bool(getattr(args, "observation_focused_state", False))

        for idx, dial in enumerate(self.dataset[sample_id][0]['dialog']):
            current_turn = self._dialog_turn

            context_memories = memory_bank.context_retrieve(dial['SPEAKER_1'], n_results=self.context_memory_number, current_turn=current_turn, datatype='text')
            related_memories = memory_bank.relevance_retrieve(dial['SPEAKER_1'], n_results=self.relevance_memory_number, dist_thres=self.dist_thres, current_turn=current_turn, datatype='text')

            if len(related_memories) == 0:
                summarized_related_memories = ["No relevant Memories."]
            else:
                summarized_related_memories = []
                for related_memory in related_memories:
                    mem_turn = int(related_memory.get('turn', 0))
                    ago = format_turns_ago(current_turn - mem_turn)
                    summarized_related_memories.append(f"{ago} ago, " + f"{related_memory['summary']}.")

            if len(context_memories) == 0:
                agent_context = []
                agent_context.append(f"In this turn, {self.usr_name} said: {dial['SPEAKER_1']}.")

            else:
                agent_context = [f"[TURN {context_memory['idx']}]" + f" : {context_memory['dialog']}." for context_memory in context_memories]
                agent_context.append(f"In this turn, {self.usr_name} said: {dial['SPEAKER_1']}.")

            # Warmup persona building: let FEP accumulate p(o|s) and p(s'|s,a) during warmup
            if warmup_with_personas:
                therapist_action_arg = resolve_persona_therapist_action_for_traits(
                    dial, args, self.logger
                )

                # Use long-prompt predictor for warmup state (untruncated history)
                if self._state_label_space in ("cami_stage", "mi_talk"):
                    if not hasattr(self, "_lp_warmup_history") or not hasattr(self, "_lp_warmup_sample_id") or self._lp_warmup_sample_id != sample_id:
                        self._lp_warmup_history = []
                        self._lp_warmup_sample_id = sample_id
                    if self._state_label_space == "mi_talk":
                        warmup_label = self._cot_classify_mi_talk(
                            self._lp_warmup_history, dial["SPEAKER_1"], dial["SPEAKER_2"],
                        )
                    else:
                        probs = personas._observation_focused_state_call(dial["SPEAKER_1"])
                        warmup_label = max(probs, key=probs.get) if probs else None
                    if warmup_label in self._state_labels:
                        personas._latent_mi["client_talk_type"] = warmup_label
                    self._lp_warmup_history.append(f"{self.usr_name}: {dial['SPEAKER_1']}")
                    self._lp_warmup_history.append(f"{self.agent_name}: {dial['SPEAKER_2']}")

                personas.traits_update(
                    dial["SPEAKER_1"],
                    dial["SPEAKER_2"],
                    therapist_action=therapist_action_arg,
                )

                # Run planner during warmup so eval turn 0 has prior
                if not getattr(args, "disable_planner", False):
                    current_patient_state = personas._format_current_state()
                    current_observation_model = personas._format_observation_model()
                    current_transition_model = personas._format_transition_model()
                    _no_wm = bool(getattr(args, "planner_no_wm_input", False))
                    state_after = planner.predict_post_action_state(
                        current_patient_state,
                        "" if _no_wm else current_observation_model,
                        "" if _no_wm else current_transition_model,
                        action=therapist_action_arg,
                        therapist_utterance=dial["SPEAKER_2"],
                        therapist_actions=personas.therapist_actions,
                    )
                    # Extract next-state prediction and set as prior for next turn
                    next_pred = (
                        _extract_cami_stage_from_text(state_after)
                        if self._state_label_space == "cami_stage"
                        else _extract_mi_talk_type_from_text(state_after)
                    )
                    personas.set_planner_prior(next_pred)

                if self.logger:
                    self.logger.info(
                        "[MSC] warmup traits_update sample=%s turn=%s obs_rules=%d trans_rules=%d state=%s",
                        sample_id, idx,
                        len(personas._observation_model_block),
                        len(personas._transition_model_block),
                        personas._latent_mi.get("client_talk_type", "?"),
                    )

            response_data = {"idx": len(memory_bank.short_term_memory), "turn": current_turn, "dialog": f"SPEAKER_2: {dial['SPEAKER_2']}"}
            memory_bank.short_term_memory.append(response_data)
            self._dialog_turn += 1

        return memory_bank, personas, response_generator, planner


    def compute_scores(self, response, calculate_dist_n, reference_text):
        metrics_dict = calc_nlg_metrics([response], [reference_text], "response")

        bleu_score = np.array([metrics_dict['Bleu_1'], metrics_dict['Bleu_2'], metrics_dict['Bleu_3'], metrics_dict['Bleu_4']])
        rl_score = np.array([metrics_dict['ROUGE_L']])
        dist_score = np.array([calculate_dist_n(response, i) for i in [1, 2, 3]])

        score = np.concatenate((bleu_score, rl_score, dist_score))

        return score

    def predict_current_state_label(self, personas, current_patient_state):
        """
        Return the current-turn patient state label, derived from `personas.q_turn_state_probs` or from text.

        --state_predict_from_probs=True (default):
            Use the argmax of `q_turn_state_probs` uniformly (when persona_state_space and
            state_label_space agree, the probs are already in the right label space); otherwise fall back to text parsing.
        --state_predict_from_probs=False:
            Legacy behavior: mi_talk uses probs, cami_stage uses text parsing.
        """
        use_probs = getattr(self.args, "state_predict_from_probs", True)
        probs = personas.q_turn_state_probs or {}

        if use_probs or self._state_label_space == "mi_talk":
            # Try to take argmax over the probability distribution.
            best_label = None
            best_score = None
            for label in self._state_labels:
                score = float(probs.get(label, 0.0))
                if best_score is None or score > best_score:
                    best_label = label
                    best_score = score
            if best_label is not None:
                if self.logger:
                    self.logger.debug(
                        "[MSC] predict_current_state: probs_argmax=%s score=%.3f probs=%s",
                        best_label, best_score, probs,
                    )
                return best_label

        # Fall back to text parsing.
        if self._state_label_space == "mi_talk":
            return _extract_mi_talk_type_from_text(current_patient_state)
        return _extract_cami_stage_from_text(current_patient_state)

    def resolve_next_state_eval_mode(self, next_gold_state):
        mode = getattr(self.args, "next_state_eval_mode", "auto")
        if mode == "off":
            return "off"
        if mode == "label":
            return "label" if next_gold_state is not None else "off"
        if mode == "alignment":
            return "alignment"
        if next_gold_state is not None:
            return "label"
        return "alignment"

    def eval_state_alignment(self, user_traits, next_patient_utt, doctor_reply):
        """
        Evaluation: agreement between the structured next-state prediction and the client's actual next-turn utterance (LLM-rated, 0-1).
        """
        user_prompt = format_state_alignment_user(
            self.agent_name, doctor_reply, user_traits, next_patient_utt
        )
        raw = self.client.employ(STATE_ALIGNMENT_SYS, user_prompt, "ResponseGenerator")
        data = _extract_json_object(raw) or {}
        try:
            v = float(data.get("alignment_score", 0.5))
        except (TypeError, ValueError):
            v = 0.5
        return max(0.0, min(1.0, v))

    def eval_next_state_metric(
        self,
        next_state_text,
        next_gold_state,
        next_patient_utt,
        doctor_reply,
    ):
        resolved_mode = self.resolve_next_state_eval_mode(next_gold_state)
        if resolved_mode == "off":
            pred = (
                _extract_cami_stage_from_text(next_state_text)
                if self._state_label_space == "cami_stage"
                else _extract_mi_talk_type_from_text(next_state_text)
            )
            return None, "off", pred

        predicted_next_state = (
            _extract_cami_stage_from_text(next_state_text)
            if self._state_label_space == "cami_stage"
            else _extract_mi_talk_type_from_text(next_state_text)
        )
        if resolved_mode == "label":
            if next_gold_state is None or predicted_next_state is None:
                return None, "label", predicted_next_state
            return float(predicted_next_state == next_gold_state), "label", predicted_next_state

        if not next_patient_utt:
            return None, "alignment", predicted_next_state
        score = self.eval_state_alignment(next_state_text, next_patient_utt, doctor_reply)
        return score, "alignment", predicted_next_state

    def eval_next_state_label_metric(self, predicted_next_state, next_gold_state):
        resolved_mode = self.resolve_next_state_eval_mode(next_gold_state)
        if resolved_mode != "label":
            return None, resolved_mode
        if next_gold_state is None or predicted_next_state is None:
            return None, resolved_mode
        return float(predicted_next_state == next_gold_state), resolved_mode


    _COT_MI_TALK_PROMPT = (
        "During the Motivational Interviewing counseling conversation, the client's utterances "
        "can be classified into the following talk types:\n\n"
        "- **neutral**: The client is mainly informative, relational, or factual, without clear "
        "movement toward or against change. Includes backchannels (\"yeah\", \"mm-hmm\"), "
        "factual descriptions, and mixed/ambiguous statements.\n"
        "- **change**: The client expresses desire, ability, reason, need, commitment, or steps "
        "toward change. Key markers: \"I want to\", \"I should\", expressing dissatisfaction "
        "with status quo, naming consequences.\n"
        "- **sustain**: The client actively defends the status quo, minimizes the problem, resists "
        "change, or argues for staying the same. Key markers: \"I don't see the problem\", "
        "\"it's not that bad\", rationalizing current behavior.\n\n"
        "Given the current counseling context, analyze the context step by step and then infer "
        "the talk type of the client's LAST utterance. "
        "**Your response should end with "
        "\"Therefore, the client's current talk type in the above context is [talk_type]\".**\n\n"
    )

    def _cot_classify_mi_talk(self, history_before, patient_utt, therapist_utt):
        """MI talk-type classification from dialogue context.
        Only includes patient utterance (not doctor response) to avoid bias."""
        full = list(history_before) + [
            f"{self.usr_name}: {patient_utt}",
        ]
        max_chars = int(getattr(self.args, "state_history_max_chars", 16000) or 16000)
        conv_text = "\n".join(
            s.replace(self.usr_name + ":", "- Client:").replace(self.agent_name + ":", "- Counselor:")
            for s in full
        )
        if len(conv_text) > max_chars:
            conv_text = "(history truncated)\n" + conv_text[-max_chars:]

        prompt = self._COT_MI_TALK_PROMPT + f"### Dialogue Context\n{conv_text}\n\n### Talk Type Inference\n"
        raw = self.client.employ(
            "You are an expert MI (Motivational Interviewing) talk-type classifier.",
            prompt,
            "PersonaExtraction",
            temperature=0.1,
            top_p=0.1,
            max_tokens=512,
        )
        raw_lower = (raw or "").lower()
        # Extract label from "talk type ... is [label]"
        import re as _re
        m = _re.search(r"talk\s*type\s+(?:in\s+the\s+above\s+context\s+)?is\s+\[?(\w+)\]?", raw_lower)
        if m and m.group(1) in MI_TALK_TYPES:
            return m.group(1)
        # Fallback: last occurrence of a label
        for label in ("change", "sustain", "neutral"):
            if label in raw_lower[-150:]:
                return label
        return "neutral"

    def interative_eval(self, sample_id, session_num, memory_bank, personas, response_generator, planner, args,
                        miti_scorer=None):
        """
        Interactive evaluation on later sessions (only memory warmup is done relative to memory_bank_init).
        Per turn: update the persona with the current patient utterance P_i and therapist utterance D_i; TransitionPlanner does one p(s'|s,a) roll-forward given the full user model and action a;
        the client simulator sees both USER_MODEL and STATE_AFTER_THERAPIST_ACTION; compare against the ground-truth P_{i+1} via NLG; state-alignment scoring prefers the planner output (falling back to user_traits when no planner).
        The final turn (no P_{i+1}) is excluded from the generation and state metric denominators.
        miti_scorer: optional MITIScorer instance; when provided, classifies each counselor utterance.
        """
        dialog = self.dataset[sample_id][session_num]['dialog']
        n_dialog = len(dialog)

        # 8-dim NLG metrics + current-state accuracy + next-state evaluation (label-acc, or alignment fallback).
        sum_score = np.zeros(len(METRIC_NAMES))
        count_score = np.zeros(len(METRIC_NAMES))

        for idx, dial in enumerate(dialog):
            current_turn = self._dialog_turn
            utter_sum_nlg = np.zeros(8)
            next_patient = dialog[idx + 1]['SPEAKER_1'] if idx + 1 < n_dialog else None
            current_gold_state = dial.get(self._state_key)
            if current_gold_state is not None:
                current_gold_state = str(current_gold_state).strip().lower()
                if current_gold_state not in self._state_labels:
                    current_gold_state = None
            next_gold_state = (
                dialog[idx + 1].get(self._state_key)
                if idx + 1 < n_dialog
                else None
            )
            if next_gold_state is not None:
                next_gold_state = str(next_gold_state).strip().lower()
                if next_gold_state not in self._state_labels:
                    next_gold_state = None

            # Record the turn's dialogue ID and raw content (for downstream alignment).
            turn_id = {
                "sample_id": int(sample_id),
                "session_num": int(session_num),
                "turn_idx": int(idx),
            }
            self.logger.info(
                "[MSC] turn_id=%s patient_utt=%r doctor_utt=%r",
                turn_id,
                dial.get("SPEAKER_1", "")[:240],
                dial.get("SPEAKER_2", "")[:240],
            )

            # Short-term context (what the patient just said) + relevant memory retrieved with the current therapist utterance as query.
            context_memories = memory_bank.context_retrieve(
                dial['SPEAKER_1'], n_results=self.context_memory_number, current_turn=current_turn, datatype='text'
            )
            related_memories = memory_bank.relevance_retrieve(
                dial['SPEAKER_2'], n_results=self.relevance_memory_number, dist_thres=self.dist_thres, current_turn=current_turn, datatype='text'
            )

            if len(related_memories) == 0:
                summarized_related_memories = ["No relevant Memories."]
            else:
                summarized_related_memories = []
                for related_memory in related_memories:
                    mem_turn = int(related_memory.get('turn', 0))
                    ago = format_turns_ago(current_turn - mem_turn)
                    summarized_related_memories.append(f"{ago} ago, " + f"{related_memory['summary']}.")

            if len(context_memories) == 0:
                context = [f"In this turn, {self.usr_name} said: {dial['SPEAKER_1']}."]
            else:
                context = [
                    f"[TURN {context_memory['idx']}] : {context_memory['dialog']}."
                    for context_memory in context_memories
                ]
                context.append(f"In this turn, {self.usr_name} said: {dial['SPEAKER_1']}.")

            merged_relevant_memory = "\n".join(summarized_related_memories)
            merged_context = "\n".join(context)

            therapist_action_arg = resolve_persona_therapist_action_for_traits(
                dial, args, self.logger
            )

            # --- State classification: use long-prompt when observation_focused_state ---
            use_longprompt_q = bool(
                getattr(args, "observation_focused_state", False)
            )

            if use_longprompt_q:
                # PUMA state head uses the local inference implementation.
                if not hasattr(self, "_lp_history") or self._lp_sample_id != sample_id:
                    # Reset history for each new sample (avoid cross-sample contamination)
                    self._lp_sample_id = sample_id
                    self._lp_history = []
                    for d in self.dataset[sample_id][0]["dialog"]:
                        self._lp_history.append(f"{self.usr_name}: {d['SPEAKER_1']}")
                        self._lp_history.append(f"{self.agent_name}: {d['SPEAKER_2']}")

                # Step 1: state classification using the local PUMA state head.
                if self._state_label_space == "mi_talk":
                    lp_label = self._cot_classify_mi_talk(
                        self._lp_history, dial["SPEAKER_1"], dial["SPEAKER_2"],
                    )
                else:
                    probs = personas._observation_focused_state_call(dial["SPEAKER_1"])
                    lp_label = max(probs, key=probs.get) if probs else None

                # Step 2: planner prior refinement (only when prior is available and disagrees)
                planner_prior = personas._planner_prior
                final_label = lp_label
                refinement_info = "no_prior"
                _fusion_mode = getattr(args, "prior_fusion_mode", "adjudicate")

                if planner_prior is not None and lp_label and _fusion_mode != "none":
                    prior_label = max(planner_prior, key=planner_prior.get)
                    if prior_label != lp_label:
                        # Build label description for refinement prompts
                        if self._state_label_space == "cami_stage":
                            _label_desc = (
                                "Use ONLY these labels: precontemplation, contemplation, preparation.\n"
                                "- precontemplation: does not see the behavior as a problem, defends status quo.\n"
                                "- contemplation: recognizes some issue but is ambivalent, mixed signals.\n"
                                "- preparation: ready to act, concrete steps or commitments.\n"
                            )
                        else:
                            _label_desc = (
                                "Use ONLY these labels: neutral, change, sustain.\n"
                                "- neutral: factual or non-committal, no clear direction toward or against change.\n"
                                "- change: expresses desire, ability, reason, need, or commitment to change.\n"
                                "- sustain: defends status quo, minimizes problem, or resists change.\n"
                            )

                        if _fusion_mode == "adjudicate":
                            # --- Variant 1: fair LLM judge with 2-turn context ---
                            # Give LLM only the last 2 turns (not full history) so it can't
                            # just repeat the long-prompt prediction; it must judge from the
                            # patient's actual words which prediction is more plausible.
                            recent_ctx = ""
                            if hasattr(self, "_lp_history") and self._lp_history:
                                tail = self._lp_history[-4:]  # up to 2 prior turns (patient+doctor each)
                                recent_ctx = "\n".join(tail)
                            refine_sys = (
                                "You are a neutral patient state judge. "
                                "Two classifiers have given different predictions for the patient's current state. "
                                "Your job is to decide which prediction is more consistent with the patient's actual words.\n"
                                + _label_desc +
                                "Return strict JSON."
                            )
                            refine_user = (
                                f"Recent dialogue context (last 2 turns):\n{recent_ctx}\n\n"
                                f"Patient's current utterance:\n{dial['SPEAKER_1']}\n\n"
                                f"Classifier A (full dialogue history): **{lp_label}**\n"
                                f"Classifier B (FEP world model, based on previous state + therapist action): **{prior_label}**\n\n"
                                "Judge which prediction better matches the patient's current utterance. "
                                "Focus on the patient's words, not on which classifier sounds more authoritative.\n\n"
                                'Output JSON: {"current_state":"<label>","rationale":"<one sentence>"}'
                            )
                            import re as _re
                            raw_refine = self.client.employ(refine_sys, refine_user, "PriorRefinement")
                            _m = _re.search(r'"current_state"\s*:\s*"(\w+)"', raw_refine or "")
                            if _m and _m.group(1).lower() in self._state_labels:
                                final_label = _m.group(1).lower()

                        elif _fusion_mode == "soft_fusion":
                            # --- Variant 2: soft weighted fusion, no extra LLM call ---
                            # alpha = lp confidence, derived from utterance surface features
                            import re as _re
                            _utt = dial["SPEAKER_1"]
                            _n = len(_utt.split())
                            _hedges = {"maybe", "i guess", "i don't know", "kind of", "sort of",
                                       "mm", "hmm", "yeah", "okay", "right", "uh", "um",
                                       "not sure", "i mean", "well"}
                            _has_hedge = any(h in _utt.lower() for h in _hedges)
                            if _n < 6:
                                _alpha = 0.50
                            elif _n < 12 or _has_hedge:
                                _alpha = 0.65
                            elif _n < 25:
                                _alpha = 0.75
                            else:
                                _alpha = 0.85
                            _beta = float(getattr(args, "fep_soft_fusion_beta", 0.35))
                            _labels = list(self._state_labels)
                            _lp_probs = {k: (_alpha if k == lp_label else (1 - _alpha) / (len(_labels) - 1))
                                         for k in _labels}
                            _fused = {k: (1 - _beta) * _lp_probs[k] + _beta * planner_prior.get(k, 0.15)
                                      for k in _labels}
                            final_label = max(_fused, key=_fused.get)
                            refinement_info = (f"prior={prior_label} lp={lp_label} final={final_label} "
                                               f"alpha={_alpha} beta={_beta}")

                        elif _fusion_mode == "llm_informed":
                            # --- Variant 3: LLM reasons about utterance informativeness first ---
                            import re as _re
                            refine_sys = (
                                "You are a patient state classifier with access to two predictions. "
                                "Before choosing, you must explicitly reason about how informative the patient's "
                                "current utterance is. Only then decide which prediction to trust.\n"
                                + _label_desc +
                                "Return strict JSON."
                            )
                            refine_user = (
                                f"Patient's current utterance:\n\"{dial['SPEAKER_1']}\"\n\n"
                                f"Prediction A — full dialogue history classifier: **{lp_label}**\n"
                                f"Prediction B — FEP world model (previous state + therapist action): **{prior_label}**\n\n"
                                "Step 1: Assess utterance informativeness. Is the patient's utterance specific and "
                                "substantive (clearly indicating a readiness stage), or is it short, ambiguous, "
                                "or backchannel-like (giving little direct signal)? Explain briefly.\n"
                                "Step 2: Based on your assessment, choose the prediction that is better grounded. "
                                "If the utterance is informative, trust the dialogue-based prediction. "
                                "If ambiguous, the world model may be more reliable.\n\n"
                                'Output JSON: {"informativeness":"high|medium|low",'
                                '"current_state":"<label>","rationale":"<1-2 sentences>"}'
                            )
                            raw_refine = self.client.employ(refine_sys, refine_user, "PriorRefinement")
                            _m = _re.search(r'"current_state"\s*:\s*"(\w+)"', raw_refine or "")
                            if _m and _m.group(1).lower() in self._state_labels:
                                final_label = _m.group(1).lower()

                        if _fusion_mode != "soft_fusion":
                            refinement_info = f"prior={prior_label} lp={lp_label} final={final_label}"
                    else:
                        refinement_info = f"prior={prior_label} agrees with lp={lp_label}"

                current_state_pred = final_label

                # Feed the state back into Personas so p filter uses it
                personas._latent_mi["client_talk_type"] = current_state_pred or personas._latent_mi.get("client_talk_type", "contemplation")

                # Run p filter (for observation/transition rule updates)
                current_patient_state, current_observation_model, current_transition_model = personas.traits_update(
                    dial["SPEAKER_1"],
                    dial["SPEAKER_2"],
                    therapist_action=therapist_action_arg,
                )
                current_state_probs = {current_state_pred: 0.9} if current_state_pred else {}

                # Append to long-prompt history (untruncated)
                self._lp_history.append(f"{self.usr_name}: {dial['SPEAKER_1']}")
                self._lp_history.append(f"{self.agent_name}: {dial['SPEAKER_2']}")

                self._static_log_info(
                    "[MSC] q_s label_space=cami_stage mode=long_prompt+prior predicted_current_state=%s gold_current_state=%s refinement=%s",
                    current_state_pred, current_gold_state, refinement_info,
                )
            else:
                # Original FEP q filter
                current_patient_state, current_observation_model, current_transition_model = personas.traits_update(
                    dial["SPEAKER_1"],
                    dial["SPEAKER_2"],
                    therapist_action=therapist_action_arg,
                )
                current_state_probs = personas.q_turn_state_probs
                current_state_pred = self.predict_current_state_label(personas, current_patient_state)
            if self._state_label_space == "cami_stage":
                self._static_log_info(
                    "[MSC] q_s label_space=cami_stage predicted_current_state=%s gold_current_state=%s "
                    "q_manifold_mi_probs=%s action=%s",
                    current_state_pred,
                    current_gold_state,
                    current_state_probs,
                    therapist_action_arg,
                )
            else:
                self._static_log_info(
                    "[MSC] q_s label_space=mi_talk predicted_current_state=%s gold_current_state=%s probs=%s action=%s",
                    current_state_pred,
                    current_gold_state,
                    current_state_probs,
                    therapist_action_arg,
                )

            if current_gold_state is not None and current_state_pred is not None:
                sum_score[8] += float(current_state_pred == current_gold_state)
                count_score[8] += 1
                self._confusion_pairs.append((current_state_pred, current_gold_state))

            if getattr(args, "disable_planner", False):
                state_after_action = current_patient_state
            else:
                _no_wm = bool(getattr(args, "planner_no_wm_input", False))
                state_after_action = planner.predict_post_action_state(
                    current_patient_state,
                    "" if _no_wm else current_observation_model,
                    "" if _no_wm else current_transition_model,
                    action=therapist_action_arg,
                    therapist_utterance=dial["SPEAKER_2"],
                    therapist_actions=personas.therapist_actions,
                )

            next_state_metric, next_state_mode, next_state_pred = self.eval_next_state_metric(
                state_after_action,
                next_gold_state,
                next_patient,
                dial['SPEAKER_2'],
            )

            # Store planner's next-state prediction as transition prior for next turn
            if getattr(args, "observation_focused_state", False):
                personas.set_planner_prior(next_state_pred)

            # Uniform logging: the turn's predicted state and the (updated) user-model summary from Personas / Planner.
            self.logger.info(
                "[MSC] turn_summary sample=%s session=%s turn=%s label_space=%s "
                "doctor=%s patient=%s "
                "pred_current=%s gold_current=%s pred_next=%s gold_next=%s "
                "current_state_excerpt=%s state_after_action_excerpt=%s",
                sample_id,
                session_num,
                idx,
                self._state_label_space,
                (dial['SPEAKER_2'] or "")[:120].replace("\n", " "),
                (dial['SPEAKER_1'] or "")[:120].replace("\n", " "),
                current_state_pred,
                current_gold_state,
                next_state_pred,
                next_gold_state,
                (current_patient_state or "")[:320].replace("\n", " "),
                (state_after_action or "")[:320].replace("\n", " "),
            )
            self._static_log_info(
                "[MSC] next_state_prediction mode=%s predicted_next_state=%s gold_next_state=%s",
                next_state_mode,
                next_state_pred,
                next_gold_state,
            )

            if next_patient is not None:
                for i in range(args.build_times):
                    self.logger.debug(f"The building turn: {i}.")
                    if args.sampling:
                        response_generator.patient_sampling(
                            dial['SPEAKER_2'],
                            next_patient,
                            merged_context,
                            merged_relevant_memory,
                            state_after_action=state_after_action,
                        )
                        response = ""
                    else:
                        response = response_generator.patient_response_build(
                            dial['SPEAKER_2'],
                            merged_context,
                            merged_relevant_memory,
                            state_after_action=state_after_action,
                        )

                    if args.generation_out:
                        self.logger.info(f"doctor_reply: {dial['SPEAKER_2']}")
                        self.logger.info(f"current_state_probs: {current_state_probs}")
                        self.logger.info(f"predicted_current_state: {current_state_pred}")
                        self.logger.info(f"gold_current_state: {current_gold_state}")
                        if state_after_action:
                            self.logger.info(
                                f"planner_roll_forward: {state_after_action[:800]}{'…' if len(state_after_action) > 800 else ''}"
                            )
                        self.logger.info(f"predicted_next_state: {next_state_pred}")
                        self.logger.info(f"gold_next_state: {next_gold_state}")
                        self.logger.info(f"generated_patient: {response}")
                        self.logger.info(f"reference_next_patient: {next_patient}")

                    utter_sum_nlg += self.compute_scores(response, self.calculate_dist_n, next_patient)

                utter_avg_nlg = utter_sum_nlg / args.build_times
                sum_score[:8] += utter_avg_nlg
                count_score[:8] += 1
                if next_state_metric is not None:
                    sum_score[9] += next_state_metric
                    count_score[9] += 1

            # MITI: classify this counselor utterance and update word counts
            if miti_scorer is not None:
                miti_label = miti_scorer.add_turn(dial['SPEAKER_2'], dial['SPEAKER_1'])
                self.logger.debug(
                    "[MITI] turn=%s counselor_label=%s utt=%r",
                    idx, miti_label, (dial['SPEAKER_2'] or "")[:60],
                )

            response_data = {
                "idx": len(memory_bank.short_term_memory),
                "turn": current_turn,
                "dialog": f"SPEAKER_2: {dial['SPEAKER_2']}",
            }
            memory_bank.short_term_memory.append(response_data)
            self._dialog_turn += 1

        avg_score = np.full(len(METRIC_NAMES), np.nan)
        np.divide(sum_score, count_score, out=avg_score, where=count_score > 0)

        return avg_score, memory_bank, personas, response_generator, planner


    def calculate_dist_n(self, text, n):
        words = text.split()
        n_grams = list(ngrams(words, n))
        unique_n_grams = len(set(n_grams))
        dist_n = unique_n_grams / len(n_grams) if n_grams else 0

        return dist_n

    def evaluation(self):
        all_samples_score = []
        all_sample_avg = []
        self._confusion_pairs = []  # reset for this evaluation run
        self._miti_per_sample = []  # reset MITI tracking
        min_s, max_s = self._session_range()
        n_dataset = len(self.dataset)
        for idx, sample in enumerate(self.dataset):
            all_sessions_score = []
            # MITI: reset scorer for each sample
            if self._miti_enabled:
                self._miti_scorer.reset()
            self._dialog_turn = 0
            memory_bank, personas, response_generator, planner = self.memory_bank_init(idx, self.args)
            for session_num, i in enumerate(range(min_s, max_s)):
                score, memory_bank, personas, response_generator, planner = self.interative_eval(
                    idx, i, memory_bank, personas, response_generator, planner, self.args,
                    miti_scorer=self._miti_scorer if self._miti_enabled else None,
                )
                all_sessions_score.append(score)
            del memory_bank, personas, response_generator, planner
            avg_score = _safe_nanmean(all_sessions_score)
            all_sample_avg.append(avg_score)
            self.logger.info(f"Average Sample {idx} Result:")
            self.logger.info(_format_metric_triplets(avg_score))

            # Collect MITI scores for this sample
            if self._miti_enabled:
                sample_miti = self._miti_scorer.compute_scores()
                self._miti_per_sample.append(sample_miti)
                self.logger.info(
                    "[MITI] Sample %s scores: R/Q=%.3f  %%OQ=%.1f  %%CR=%.1f  %%MIC=%.1f  %%TTT=%.1f",
                    idx,
                    sample_miti.get("R_Q") or 0.0,
                    sample_miti.get("pct_OQ") or 0.0,
                    sample_miti.get("pct_CR") or 0.0,
                    sample_miti.get("pct_MIC") or 0.0,
                    sample_miti.get("pct_TTT") or 0.0,
                )

            self.logger.info(f"Sample {idx} Completed!")
            all_samples_score.append(all_sessions_score)

            if n_dataset > 1 and (idx + 1) % self.args.log_step == 0 and (idx + 1) < n_dataset:
                self.logger.info(f"The Average Results of {idx + 1} Samples:")
                for session_number in range(min_s + 1, max_s + 1):
                    try:
                        score_session = [sample[session_number - min_s - 1] for sample in all_samples_score]
                        avg_score = _safe_nanmean(score_session)

                        self.logger.info(f"Session {session_number} Results:")
                        self.logger.info(_format_metric_triplets(avg_score))

                    except Exception:
                        self.logger.info(f"session_number {session_number} is out of list index.")

        if self.args.sampling:
            self.sampling_file = os.path.join(self.args.sampling_path, self.args.sampling_file_name)
            with open(self.sampling_file, "w") as f:
                    json.dump(self.sampling_dataset, f, indent=4)

        # Print the final overall summary only once, to avoid duplication with per-sample / periodic summaries.
        last_idx = n_dataset - 1 if n_dataset else -1
        if last_idx >= 0 and n_dataset > 1:
            self.logger.info(f"The Average Results of {last_idx + 1} Samples:")
            for session_number in range(min_s + 1, max_s + 1):
                try:
                    score_session = [sample[session_number - min_s - 1] for sample in all_samples_score]
                    avg_score = _safe_nanmean(score_session)
                    self.logger.info(f"Session {session_number} Results:")
                    self.logger.info(_format_metric_triplets(avg_score))

                except Exception:
                    self.logger.info(f"session_number {session_number} is out of list index.")

        results_payload = self._build_results_payload(all_samples_score, all_sample_avg, min_s, max_s)
        self._save_results_json(results_payload)
