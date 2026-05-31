"""
S017 + S018 analysis: Per-class F1, confusion matrix, and turn-position drift.

Parses turn_summary lines from evaluation logs to extract per-turn (gold, pred) pairs,
then computes:
  - Per-class Precision / Recall / F1 for current-state and next-state
  - Confusion matrices
  - Turn-position accuracy curves (S018)

Usage:
    python analyze_ablation.py --output_dir figures/

Reads logs from the sshfs-mounted server path by default.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

TURN_SUMMARY_RE = re.compile(
    r"turn_summary\s+"
    r"sample=(?P<sample>\d+)\s+"
    r"session=(?P<session>\d+)\s+"
    r"turn=(?P<turn>\d+)\s+"
    r"label_space=(?P<label_space>\S+)\s+"
    r"doctor=.*?\s+"
    r"patient=.*?\s+"
    r"pred_current=(?P<pred_current>\S+)\s+"
    r"gold_current=(?P<gold_current>\S+)\s+"
    r"pred_next=(?P<pred_next>\S+)\s+"
    r"gold_next=(?P<gold_next>\S+)"
)

STAGES = ["precontemplation", "contemplation", "preparation"]


def parse_log(log_path: str) -> List[Dict]:
    """Parse turn_summary lines from a log file."""
    records = []
    with open(log_path, "r", errors="replace") as f:
        for line in f:
            if "turn_summary" not in line:
                continue
            m = TURN_SUMMARY_RE.search(line)
            if not m:
                continue
            records.append({
                "sample": int(m.group("sample")),
                "session": int(m.group("session")),
                "turn": int(m.group("turn")),
                "pred_current": m.group("pred_current"),
                "gold_current": m.group("gold_current"),
                "pred_next": m.group("pred_next"),
                "gold_next": m.group("gold_next"),
            })
    return records


def parse_multi_gpu_logs(log_pattern: str, n_gpus: int = 4) -> List[Dict]:
    """Parse turn_summary from multi-GPU log files (gpu0..gpu3)."""
    all_records = []
    for i in range(n_gpus):
        path = log_pattern.replace("_gpu0.", f"_gpu{i}.")
        if os.path.isfile(path):
            all_records.extend(parse_log(path))
    return all_records


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_per_class_metrics(
    records: List[Dict], pred_key: str, gold_key: str, classes: List[str]
) -> Dict:
    """Compute per-class P/R/F1 and confusion matrix."""
    confusion = Counter()
    for r in records:
        gold = r[gold_key]
        pred = r[pred_key]
        if gold in classes and pred in classes:
            confusion[(gold, pred)] += 1

    # Build confusion matrix
    matrix = {g: {p: confusion[(g, p)] for p in classes} for g in classes}

    # Per-class metrics
    per_class = {}
    for cls in classes:
        tp = confusion[(cls, cls)]
        fp = sum(confusion[(g, cls)] for g in classes if g != cls)
        fn = sum(confusion[(cls, p)] for p in classes if p != cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = sum(confusion[(cls, p)] for p in classes)
        per_class[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    # Macro averages
    macro_p = sum(m["precision"] for m in per_class.values()) / len(classes)
    macro_r = sum(m["recall"] for m in per_class.values()) / len(classes)
    macro_f1 = sum(m["f1"] for m in per_class.values()) / len(classes)

    total = sum(confusion.values())
    correct = sum(confusion[(c, c)] for c in classes)
    accuracy = correct / total if total > 0 else 0.0

    return {
        "per_class": per_class,
        "confusion_matrix": matrix,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "total": total,
    }


def compute_turn_position_accuracy(
    records: List[Dict], pred_key: str, gold_key: str
) -> Dict[int, Dict]:
    """Compute accuracy grouped by turn index."""
    by_turn = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in records:
        t = r["turn"]
        gold = r[gold_key]
        pred = r[pred_key]
        by_turn[t]["total"] += 1
        if gold == pred:
            by_turn[t]["correct"] += 1
    result = {}
    for t in sorted(by_turn.keys()):
        d = by_turn[t]
        result[t] = {
            "accuracy": d["correct"] / d["total"] if d["total"] > 0 else 0.0,
            "correct": d["correct"],
            "total": d["total"],
        }
    return result


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_confusion_matrix(matrix: Dict, classes: List[str], label: str):
    """Print confusion matrix as a formatted table."""
    print(f"\n{'='*60}")
    print(f"Confusion Matrix: {label}")
    print(f"{'='*60}")
    # Abbreviate class names
    abbr = {"precontemplation": "PreC", "contemplation": "Cont", "preparation": "Prep"}
    header = f"{'Gold↓ Pred→':>12s}" + "".join(f"{abbr.get(c, c):>8s}" for c in classes) + f"{'Total':>8s}"
    print(header)
    print("-" * len(header))
    for g in classes:
        row = f"{abbr.get(g, g):>12s}"
        total = 0
        for p in classes:
            v = matrix[g][p]
            total += v
            row += f"{v:>8d}"
        row += f"{total:>8d}"
        print(row)
    print()


def print_per_class_table(per_class: Dict, macro_f1: float, accuracy: float, label: str):
    """Print per-class P/R/F1 table."""
    print(f"\n{'='*60}")
    print(f"Per-Class Metrics: {label}")
    print(f"{'='*60}")
    abbr = {"precontemplation": "PreC", "contemplation": "Cont", "preparation": "Prep"}
    print(f"{'Class':>15s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    print("-" * 55)
    for cls in STAGES:
        if cls in per_class:
            m = per_class[cls]
            print(f"{abbr.get(cls, cls):>15s} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f} {m['support']:>10d}")
    print("-" * 55)
    print(f"{'Macro':>15s} {'':>10s} {'':>10s} {macro_f1:>10.3f}")
    print(f"{'Accuracy':>15s} {'':>10s} {'':>10s} {accuracy:>10.3f}")
    print()


def generate_latex_table(all_results: Dict[str, Dict], metric_type: str = "current") -> str:
    """Generate LaTeX table for per-class F1 comparison."""
    abbr = {"precontemplation": "PreC", "contemplation": "Cont", "preparation": "Prep"}
    pred_key = "pred_current" if metric_type == "current" else "pred_next"

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-class " + metric_type + r"-state metrics (CAMI full38).}")
    lines.append(r"\begin{tabular}{l" + "c" * 9 + r"c}")
    lines.append(r"\toprule")

    # Header
    header = r"\multirow{2}{*}{Method}"
    for cls in STAGES:
        header += r" & \multicolumn{3}{c}{" + abbr[cls] + "}"
    header += r" & Macro-F1 \\"
    lines.append(header)

    sub_header = ""
    for _ in STAGES:
        sub_header += r" & P & R & F1"
    sub_header += r" & \\"
    lines.append(sub_header)
    lines.append(r"\midrule")

    for name, result in all_results.items():
        row = name
        for cls in STAGES:
            m = result["per_class"].get(cls, {"precision": 0, "recall": 0, "f1": 0})
            row += f" & {m['precision']:.3f} & {m['recall']:.3f} & {m['f1']:.3f}"
        row += f" & {result['macro_f1']:.3f} \\\\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_turn_drift_data(all_drift: Dict[str, Dict[int, Dict]]) -> str:
    """Generate CSV data for turn-position accuracy plot."""
    all_turns = sorted(set(t for d in all_drift.values() for t in d.keys()))
    lines = ["turn," + ",".join(all_drift.keys())]
    for t in all_turns:
        vals = []
        for name in all_drift.keys():
            if t in all_drift[name]:
                vals.append(f"{all_drift[name][t]['accuracy']:.3f}")
            else:
                vals.append("")
        lines.append(f"{t}," + ",".join(vals))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SERVER_RESULTS = "<PROJECT_ROOT>/src/agent/logs/results"
SERVER_LOGS = "<PROJECT_ROOT>/src/agent/logs"

# Map of experiment name -> log file(s)
EXPERIMENTS = {
    "FEP baseline (S001)": {
        "type": "single",
        "log": os.path.join(SERVER_LOGS, "cami_fep_cami_stage_t38_20260412_20:38.log"),
    },
    "no\\_prior (S008)": {
        "type": "multi_gpu",
        "log_pattern": os.path.join(SERVER_RESULTS, "fep_no_prior_full38_a850ad1_20260417_115205_gpu0.log"),
        "n_gpus": 4,
    },
    "soft\\_fusion (S014)": {
        "type": "multi_gpu",
        "log_pattern": os.path.join(SERVER_RESULTS, "prior_soft_fusion_full38_a850ad1_20260417_161530_gpu0.log"),
        "n_gpus": 4,
    },
}


def main():
    parser = argparse.ArgumentParser(description="S017+S018 ablation analysis")
    parser.add_argument("--output_dir", type=str, default="figures/",
                        help="Directory to write output files.")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse all logs
    all_records = {}
    for name, cfg in EXPERIMENTS.items():
        if cfg["type"] == "single":
            records = parse_log(cfg["log"])
        else:
            records = parse_multi_gpu_logs(cfg["log_pattern"], cfg["n_gpus"])
        all_records[name] = records
        print(f"Parsed {name}: {len(records)} turns")

    # S017: Per-class F1 and confusion matrix
    print("\n" + "=" * 70)
    print("S017: PER-CLASS F1 / CONFUSION MATRIX (Current-State)")
    print("=" * 70)

    all_current_results = {}
    for name, records in all_records.items():
        result = compute_per_class_metrics(records, "pred_current", "gold_current", STAGES)
        all_current_results[name] = result
        print_per_class_table(result["per_class"], result["macro_f1"], result["accuracy"], f"{name} (Current-State)")
        print_confusion_matrix(result["confusion_matrix"], STAGES, f"{name} (Current-State)")

    print("\n" + "=" * 70)
    print("S017: PER-CLASS F1 / CONFUSION MATRIX (Next-State)")
    print("=" * 70)

    all_next_results = {}
    for name, records in all_records.items():
        result = compute_per_class_metrics(records, "pred_next", "gold_next", STAGES)
        all_next_results[name] = result
        print_per_class_table(result["per_class"], result["macro_f1"], result["accuracy"], f"{name} (Next-State)")
        print_confusion_matrix(result["confusion_matrix"], STAGES, f"{name} (Next-State)")

    # Generate LaTeX tables
    latex_current = generate_latex_table(all_current_results, "current")
    latex_next = generate_latex_table(all_next_results, "next")

    latex_path = os.path.join(args.output_dir, "s017_per_class_f1.tex")
    with open(latex_path, "w") as f:
        f.write("% Current-State Per-Class Metrics\n")
        f.write(latex_current)
        f.write("\n\n% Next-State Per-Class Metrics\n")
        f.write(latex_next)
    print(f"\nLaTeX tables written to {latex_path}")

    # S018: Turn-position drift curves
    print("\n" + "=" * 70)
    print("S018: TURN-POSITION ACCURACY CURVES (Current-State)")
    print("=" * 70)

    all_drift_current = {}
    for name, records in all_records.items():
        drift = compute_turn_position_accuracy(records, "pred_current", "gold_current")
        all_drift_current[name] = drift
        print(f"\n{name}:")
        for t in sorted(drift.keys()):
            d = drift[t]
            print(f"  turn {t:2d}: acc={d['accuracy']:.3f} ({d['correct']}/{d['total']})")

    all_drift_next = {}
    print("\n" + "=" * 70)
    print("S018: TURN-POSITION ACCURACY CURVES (Next-State)")
    print("=" * 70)
    for name, records in all_records.items():
        drift = compute_turn_position_accuracy(records, "pred_next", "gold_next")
        all_drift_next[name] = drift
        print(f"\n{name}:")
        for t in sorted(drift.keys()):
            d = drift[t]
            print(f"  turn {t:2d}: acc={d['accuracy']:.3f} ({d['correct']}/{d['total']})")

    # Save drift data as CSV
    csv_current = generate_turn_drift_data(all_drift_current)
    csv_next = generate_turn_drift_data(all_drift_next)
    csv_path = os.path.join(args.output_dir, "s018_turn_drift_current.csv")
    with open(csv_path, "w") as f:
        f.write(csv_current)
    csv_path2 = os.path.join(args.output_dir, "s018_turn_drift_next.csv")
    with open(csv_path2, "w") as f:
        f.write(csv_next)
    print(f"\nDrift data written to {csv_path} and {csv_path2}")

    # Save full JSON results
    json_path = os.path.join(args.output_dir, "s017_s018_analysis.json")
    output = {
        "s017_current_state": {
            name: {
                "per_class": result["per_class"],
                "confusion_matrix": result["confusion_matrix"],
                "macro_f1": result["macro_f1"],
                "accuracy": result["accuracy"],
                "total": result["total"],
            }
            for name, result in all_current_results.items()
        },
        "s017_next_state": {
            name: {
                "per_class": result["per_class"],
                "confusion_matrix": result["confusion_matrix"],
                "macro_f1": result["macro_f1"],
                "accuracy": result["accuracy"],
                "total": result["total"],
            }
            for name, result in all_next_results.items()
        },
        "s018_turn_drift_current": {
            name: {str(t): d for t, d in drift.items()}
            for name, drift in all_drift_current.items()
        },
        "s018_turn_drift_next": {
            name: {str(t): d for t, d in drift.items()}
            for name, drift in all_drift_next.items()
        },
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Full analysis written to {json_path}")


if __name__ == "__main__":
    main()
