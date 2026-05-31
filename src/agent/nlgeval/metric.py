"""
Lightweight NLGEval used by calc_nlg_metrics (the original vendor metric/ tree is not vendored here).
BLEU uses NLTK; ROUGE-L uses `rouge_score` when available, otherwise returns 0.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

try:
    from rouge_score import rouge_scorer

    _HAS_ROUGE = True
except ImportError:
    _HAS_ROUGE = False


def _tokenize(s: str) -> List[str]:
    return nltk.word_tokenize(s.lower())


class NLGEval:
    def __init__(self, no_glove: bool = False, metrics_to_omit: Sequence[str] | None = None):
        self.no_glove = no_glove
        self.metrics_to_omit = set(metrics_to_omit or [])
        self._sm = SmoothingFunction().method1
        self._rouge = (
            rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True) if _HAS_ROUGE else None
        )

    def compute_metrics(
        self,
        refs_for_hyps: Sequence[Sequence[str]],
        hyp_list: Sequence[str],
    ) -> Tuple[dict, list]:
        """
        refs_for_hyps[i] is the list of references for the i-th hypothesis (here there is only one reference per hypothesis).
        """
        agg: dict[str, float] = {
            "Bleu_1": 0.0,
            "Bleu_2": 0.0,
            "Bleu_3": 0.0,
            "Bleu_4": 0.0,
            "ROUGE_L": 0.0,
        }
        per: list = []
        n = len(hyp_list)
        if n == 0:
            return agg, per

        weights = (
            (1, 0, 0, 0),
            (0.5, 0.5, 0, 0),
            (1.0 / 3, 1.0 / 3, 1.0 / 3, 0),
            (0.25, 0.25, 0.25, 0.25),
        )
        for i, hyp in enumerate(hyp_list):
            refs = list(refs_for_hyps[i]) if i < len(refs_for_hyps) else [hyp]
            ref_toks = [_tokenize(r) for r in refs if r]
            hyp_toks = _tokenize(hyp)
            if not ref_toks:
                ref_toks = [hyp_toks]
            row: dict[str, Any] = {}
            for k in range(1, 5):
                key = f"Bleu_{k}"
                if key in self.metrics_to_omit:
                    continue
                try:
                    b = float(
                        sentence_bleu(
                            ref_toks,
                            hyp_toks,
                            weights=weights[k - 1],
                            smoothing_function=self._sm,
                        )
                    )
                except ValueError:
                    b = 0.0
                row[key] = b
                agg[key] += b
            rl = 0.0
            if self._rouge and refs:
                r = self._rouge.score(refs[0], hyp)["rougeL"].fmeasure
                rl = float(r)
            row["ROUGE_L"] = rl
            agg["ROUGE_L"] += rl
            per.append(row)

        for k in agg:
            agg[k] /= n
        return agg, per
