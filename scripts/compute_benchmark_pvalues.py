#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


METHOD_DISPLAY_NAMES = {
    "samplingpkdesign": "PKProbDesign",
    "pkprobdesign": "PKProbDesign",
    "desirna": "DesiRNA",
    "modena": "MODENA",
    "antarna_pkiss": "antaRNA",
    "antarna": "antaRNA",
}

METRICS = {
    "combined_score": {"higher_is_better": True, "label": "Thermodynamic support"},
    "hamming_distance_structure_normalized": {"higher_is_better": False, "label": "Normalized Hamming distance (folded structure)"},
    "viterbi_structure_distance_normalized": {"higher_is_better": False, "label": "Normalized Hamming distance (direct Viterbi on target-derived scaffold G)"},
    "two_stage_structure_distance_normalized": {"higher_is_better": False, "label": "Normalized Hamming distance (two-stage MFE)"},
    "ipknot_structure_distance_normalized": {"higher_is_better": False, "label": "Normalized Hamming distance (IPknot)"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute paired method-comparison p-values from benchmark summary tables.")
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=Path("results/tsubame_seqlen100/summary"),
        help="Directory containing best_by_method.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <summary-root>/pvalues_wilcoxon.csv.",
    )
    return parser.parse_args()


def display_method_label(method: str) -> str:
    return METHOD_DISPLAY_NAMES.get(method.strip().lower(), method)


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    return ranks


def wilcoxon_signed_rank(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    diffs = (pd.to_numeric(x, errors="coerce") - pd.to_numeric(y, errors="coerce")).dropna()
    diffs = diffs[diffs != 0]
    n = int(diffs.shape[0])
    if n == 0:
        return float("nan"), float("nan"), 0
    abs_diffs = [abs(float(v)) for v in diffs.tolist()]
    ranks = average_ranks(abs_diffs)
    w_plus = sum(rank for rank, diff in zip(ranks, diffs.tolist(), strict=False) if diff > 0)
    w_minus = sum(rank for rank, diff in zip(ranks, diffs.tolist(), strict=False) if diff < 0)
    mean_w = n * (n + 1) / 4.0
    tie_counts: dict[float, int] = {}
    for value in abs_diffs:
        tie_counts[value] = tie_counts.get(value, 0) + 1
    tie_correction = sum(t * (t + 1) * (2 * t + 1) for t in tie_counts.values() if t > 1) / 48.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0 - tie_correction
    if var_w <= 0:
        return w_plus, 1.0, n
    continuity = 0.5 if w_plus > mean_w else -0.5 if w_plus < mean_w else 0.0
    z = (w_plus - mean_w - continuity) / math.sqrt(var_w)
    # erfc evaluates the two-sided normal tail directly.  Computing
    # ``1 - normal_cdf(abs(z))`` loses all precision for strong effects.
    p_value = math.erfc(abs(z) / math.sqrt(2.0))
    p_value = max(0.0, min(1.0, p_value))
    return w_plus, p_value, n


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    m = len(indexed)
    adjusted = [1.0] * m
    running = 1.0
    for rank, (idx, p_val) in enumerate(reversed(indexed), start=1):
        k = m - rank + 1
        candidate = min(running, p_val * m / k)
        adjusted[idx] = candidate
        running = candidate
    return adjusted


def main() -> int:
    args = parse_args()
    output_path = args.output or (args.summary_root / "pvalues_wilcoxon.csv")
    best_path = args.summary_root / "best_by_method.csv"
    best = pd.read_csv(best_path)
    best = best.copy()
    best["display_method"] = best["method"].fillna("").astype(str).map(display_method_label)
    best = best.drop_duplicates(subset=["pkb_id", "display_method"], keep="first")

    rows: list[dict[str, object]] = []
    for metric, meta in METRICS.items():
        if metric not in best.columns:
            continue
        pivot = best.pivot(index="pkb_id", columns="display_method", values=metric)
        methods = [col for col in pivot.columns if str(col).strip()]
        for i, method_a in enumerate(methods):
            for method_b in methods[i + 1:]:
                pair = pivot[[method_a, method_b]].dropna()
                if pair.empty:
                    continue
                x = pd.to_numeric(pair[method_a], errors="coerce")
                y = pd.to_numeric(pair[method_b], errors="coerce")
                if not meta["higher_is_better"]:
                    x = -x
                    y = -y
                statistic, p_value, n = wilcoxon_signed_rank(x, y)
                delta = (x - y).median()
                rows.append(
                    {
                        "metric": metric,
                        "metric_label": meta["label"],
                        "method_a": method_a,
                        "method_b": method_b,
                        "paired_targets": n,
                        "wilcoxon_w_plus": statistic,
                        "p_value": p_value,
                        "median_advantage_method_a": delta,
                        "better_direction": "higher" if meta["higher_is_better"] else "lower",
                    }
                )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_value_fdr_bh"] = benjamini_hochberg(result["p_value"].fillna(1.0).tolist())
        result = result.sort_values(["metric", "p_value", "method_a", "method_b"], na_position="last").reset_index(drop=True)
    result.to_csv(output_path, index=False)
    print(f"Wrote {len(result)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
