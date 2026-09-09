#!/usr/bin/env python3
"""Aggregate the Supplementary common-G_small four-method sensitivity analysis."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aggregate_common_gbig_four_method import add_ipknot_distances, merge_stage, save
from compute_benchmark_pvalues import wilcoxon_signed_rank
from select_common_gbig_representatives import select_representatives


METHOD_ORDER = ("pkprobdesign", "desirna", "modena_ipknot", "antarna_pkiss")
METHOD_LABELS = {
    "pkprobdesign": "PKProbDesign",
    "desirna": "DesiRNA",
    "modena_ipknot": "MODENA (IPknot)",
    "antarna_pkiss": "antaRNA (pKiss)",
}
COLORS = {
    "pkprobdesign": "#4c72b0",
    "desirna": "#dd8452",
    "modena_ipknot": "#55a868",
    "antarna_pkiss": "#8172b2",
}
PROBABILITY_COLUMNS = ("logP_G", "logP_G_prime_given_G_S", "combined_score")
STAGES = ("two_stage_mfe", "hotknots_viterbi", "ipknot_supplementary")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--common-gbig-final-records", type=Path, required=True)
    parser.add_argument("--common-gbig-probability-records", type=Path, required=True)
    parser.add_argument("--expected-targets", type=int, default=254)
    return parser.parse_args()


def read_target_ids(path: Path) -> list[str]:
    frame = pd.read_csv(path, usecols=["pkb_id"])
    ids = frame["pkb_id"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("target manifest contains duplicate IDs")
    return ids


def read_stage_tables(
    root: Path,
    stage: str,
    target_ids: list[str],
    *,
    variant: str = "small",
    expected_rows_per_target: int = 4,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for pkb_id in target_ids:
        path = root / "eval" / stage / pkb_id / variant / "candidates.csv"
        if not path.is_file():
            raise ValueError(f"missing {stage} table: {path}")
        frame = pd.read_csv(path, low_memory=False)
        if len(frame) != expected_rows_per_target:
            raise ValueError(
                f"{pkb_id}/{stage}: expected {expected_rows_per_target} rows, "
                f"found {len(frame)}"
            )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def classify_probability_pool(
    frame: pd.DataFrame, expected_targets: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    expected_rows = expected_targets * len(METHOD_ORDER) * 20
    if len(frame) != expected_rows or frame["pkb_id"].nunique() != expected_targets:
        raise ValueError("probability pool target or row count mismatch")
    if set(frame["method"]) != set(METHOD_ORDER):
        raise ValueError("probability pool method set mismatch")
    counts = frame.groupby(["pkb_id", "method"]).size()
    if len(counts) != expected_targets * len(METHOD_ORDER) or not counts.eq(20).all():
        raise ValueError("probability pool must contain exactly 20 candidates per method/target")
    if frame["candidate_uid"].nunique() != expected_rows:
        raise ValueError("probability pool candidate UIDs are not globally unique")
    if frame.duplicated(["pkb_id", "method", "rank"]).any():
        raise ValueError("probability pool target/method/rank identities are not unique")

    errors = frame["probability_error"].fillna("").astype(str).str.strip()
    error_rows = frame.loc[errors.ne("")].copy()
    if len(error_rows):
        error_rows["implementation_error_reason"] = errors.loc[errors.ne("")]

    nonfinite_rows: list[dict[str, object]] = []
    invalid_rows: list[str] = []
    for column in PROBABILITY_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for row in frame.itertuples(index=False):
            value = getattr(row, column)
            raw_error = getattr(row, "probability_error", "")
            error = "" if pd.isna(raw_error) else str(raw_error).strip()
            if value == -math.inf and not error:
                nonfinite_rows.append(
                    {
                        "pkb_id": row.pkb_id,
                        "method": row.method,
                        "rank": row.rank,
                        "candidate_uid": row.candidate_uid,
                        "term": column,
                        "value": "-inf",
                        "classification": "genuine_zero_probability",
                        "reason": "evaluator_returned_negative_infinity_without_error",
                    }
                )
            elif not math.isfinite(value) and not error:
                invalid_rows.append(f"{row.pkb_id}/{row.method}/{row.rank}/{column}")
        if frame[column].gt(1e-8).any():
            raise ValueError(f"probability pool {column} contains values greater than zero")
    if invalid_rows:
        raise ValueError(
            "probability pool contains unexplained NaN/+inf values: "
            + ", ".join(invalid_rows[:5])
        )
    nonfinite = pd.DataFrame(
        nonfinite_rows,
        columns=(
            "pkb_id",
            "method",
            "rank",
            "candidate_uid",
            "term",
            "value",
            "classification",
            "reason",
        ),
    )
    return frame, nonfinite, error_rows


def validate_selected(frame: pd.DataFrame, expected_targets: int) -> pd.DataFrame:
    counts = frame.groupby(["pkb_id", "method"]).size()
    if (
        frame["pkb_id"].nunique() != expected_targets
        or set(frame["method"]) != set(METHOD_ORDER)
        or len(counts) != expected_targets * len(METHOD_ORDER)
        or not counts.eq(1).all()
    ):
        raise ValueError("selected table must contain one row per target and method")
    if frame["candidate_uid"].duplicated().any():
        raise ValueError("selected candidate UIDs are not unique")
    if not frame["selection_uses_target_distance"].astype(str).str.lower().eq("false").all():
        raise ValueError("representative selection was not declared distance-independent")
    errors = frame["probability_error"].fillna("").astype(str).str.strip()
    if errors.ne("").any():
        raise ValueError("selected rows contain probability evaluation errors")
    for column in PROBABILITY_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        values = frame[column].to_numpy(float)
        if np.isnan(values).any() or np.isposinf(values).any():
            raise ValueError(f"selected {column} contains NaN or +inf")
    return frame


def validate_representative_selection(
    probability: pd.DataFrame, selected: pd.DataFrame
) -> None:
    recomputed: set[str] = set()
    for _, target in probability.groupby("pkb_id", sort=False):
        rows = select_representatives(
            target.fillna("").astype(str).to_dict("records"),
            per_method=20,
            analysis_label="common-G_small",
        )
        recomputed.update(row["candidate_uid"] for row in rows)
    if recomputed != set(selected["candidate_uid"]):
        raise ValueError("selected representatives do not reproduce from probability-only rule")


def finite_summary(series: pd.Series) -> dict[str, object]:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric[np.isfinite(numeric)]
    return {
        "finite_n": len(finite),
        "negative_infinity_n": int(np.isneginf(numeric).sum()),
        "mean_finite": finite.mean() if len(finite) else np.nan,
        "median_finite": finite.median() if len(finite) else np.nan,
        "mean_including_negative_infinity": numeric.mean(),
        "median_including_negative_infinity": numeric.median(),
    }


def method_summary(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        selected = records.loc[records["method"].eq(method)]
        row: dict[str, object] = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "n_targets": selected["pkb_id"].nunique(),
        }
        for column, label in (
            ("combined_score", "log_pseudo_joint"),
            ("logP_G", "logP_scaffold"),
            ("logP_G_prime_given_G_S", "logP_extension_given_scaffold"),
        ):
            for key, value in finite_summary(selected[column]).items():
                row[f"{label}_{key}"] = value
        for prefix in ("two_stage", "viterbi", "ipknot"):
            distance = pd.to_numeric(selected[f"{prefix}_structure_distance"], errors="coerce")
            normalized = pd.to_numeric(
                selected[f"{prefix}_structure_distance_normalized"], errors="coerce"
            )
            row[f"{prefix}_mean_distance"] = distance.mean()
            row[f"{prefix}_median_distance"] = distance.median()
            row[f"{prefix}_mean_normalized_distance"] = normalized.mean()
            row[f"{prefix}_median_normalized_distance"] = normalized.median()
            row[f"{prefix}_exact_match_rate"] = distance.eq(0).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def equal_to_best(values: pd.Series, best: float) -> pd.Series:
    if best == -math.inf:
        return values.eq(-math.inf)
    return np.isclose(values, best, rtol=1e-12, atol=1e-12)


def method_best_counts(records: pd.DataFrame) -> pd.DataFrame:
    counters = {
        method: {"unique_best_count": 0, "tied_best_count": 0, "best_including_ties": 0}
        for method in METHOD_ORDER
    }
    targets_with_ties = 0
    for _, target in records.groupby("pkb_id", sort=False):
        best = target["combined_score"].max()
        winners = target.loc[equal_to_best(target["combined_score"], best)]
        tied = len(winners) > 1
        targets_with_ties += tied
        for method in winners["method"]:
            counters[method]["best_including_ties"] += 1
            counters[method]["tied_best_count" if tied else "unique_best_count"] += 1
    return pd.DataFrame(
        [
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                **counters[method],
                "targets_with_any_tie": targets_with_ties,
            }
            for method in METHOD_ORDER
        ]
    )


def paired_tests(records: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "combined_score": True,
        "logP_G": True,
        "logP_G_prime_given_G_S": True,
        "two_stage_structure_distance_normalized": False,
        "viterbi_structure_distance_normalized": False,
        "ipknot_structure_distance_normalized": False,
    }
    rows: list[dict[str, object]] = []
    for metric, higher_is_better in metrics.items():
        pivot = records.pivot(index="pkb_id", columns="method", values=metric)
        for method_a, method_b in itertools.combinations(METHOD_ORDER, 2):
            candidate_pair = pivot[[method_a, method_b]].apply(
                pd.to_numeric, errors="coerce"
            )
            finite_mask = np.isfinite(candidate_pair).all(axis=1)
            pair = candidate_pair.loc[finite_mask]
            advantage = pair[method_a] - pair[method_b]
            if not higher_is_better:
                advantage = -advantage
            if len(advantage):
                statistic, p_value, nonzero = wilcoxon_signed_rank(
                    advantage, pd.Series(0.0, index=advantage.index)
                )
            else:
                statistic, p_value, nonzero = np.nan, np.nan, 0
            tolerance = 1e-12
            rows.append(
                {
                    "metric": metric,
                    "higher_is_better": higher_is_better,
                    "method_a": method_a,
                    "method_b": method_b,
                    "paired_targets_total": len(candidate_pair),
                    "paired_targets_finite": len(pair),
                    "paired_targets_excluded_nonfinite": len(candidate_pair) - len(pair),
                    "method_a_wins": int((advantage > tolerance).sum()),
                    "method_a_losses": int((advantage < -tolerance).sum()),
                    "ties": int((advantage.abs() <= tolerance).sum()),
                    "mean_advantage_method_a": advantage.mean(),
                    "median_advantage_method_a": advantage.median(),
                    "wilcoxon_nonzero_pairs": nonzero,
                    "wilcoxon_w_plus": statistic,
                    "wilcoxon_method": (
                        "two-sided normal approximation with tie and continuity correction"
                    ),
                    "p_value_raw": p_value,
                }
            )
    frame = pd.DataFrame(rows)
    frame["bonferroni_six_comparison_family_size"] = frame.groupby("metric")[
        "p_value_raw"
    ].transform(lambda values: int(values.notna().sum()))
    frame["p_value_bonferroni_six_comparisons"] = frame.apply(
        lambda row: (
            min(1.0, row["p_value_raw"] * row["bonferroni_six_comparison_family_size"])
            if pd.notna(row["p_value_raw"])
            else np.nan
        ),
        axis=1,
    )
    global_size = int(frame["p_value_raw"].notna().sum())
    frame["bonferroni_global_family_size"] = global_size
    frame["p_value_bonferroni_global"] = frame["p_value_raw"].map(
        lambda value: min(1.0, value * global_size) if pd.notna(value) else np.nan
    )
    return frame


def finite_values(records: pd.DataFrame, method: str, column: str) -> np.ndarray:
    values = pd.to_numeric(
        records.loc[records["method"].eq(method), column], errors="coerce"
    ).to_numpy(float)
    return values[np.isfinite(values)]


def violin_box(ax: plt.Axes, records: pd.DataFrame, column: str, ylabel: str) -> None:
    values = [finite_values(records, method, column) for method in METHOD_ORDER]
    if any(len(value) == 0 for value in values):
        raise ValueError(f"cannot plot {column}: at least one method has no finite values")
    positions = np.arange(1, len(METHOD_ORDER) + 1)
    violin = ax.violinplot(values, positions=positions, showextrema=False)
    for body, method in zip(violin["bodies"], METHOD_ORDER):
        body.set_facecolor(COLORS[method])
        body.set_edgecolor(COLORS[method])
        body.set_alpha(0.35)
    boxes = ax.boxplot(
        values, positions=positions, widths=0.18, showfliers=False, patch_artist=True
    )
    for box, method in zip(boxes["boxes"], METHOD_ORDER):
        box.set_facecolor("white")
        box.set_edgecolor(COLORS[method])
    ax.set_xticks(
        positions,
        [METHOD_LABELS[method] for method in METHOD_ORDER],
        rotation=18,
        ha="right",
    )
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def plot_probability(records: pd.DataFrame, figures: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    violin_box(axes[0], records, "combined_score", "log pseudo-joint probability")
    violin_box(axes[1], records, "logP_G", r"log $P(G_{\mathrm{scaffold}}\mid S)$")
    violin_box(
        axes[2],
        records,
        "logP_G_prime_given_G_S",
        r"log $P(G_{\mathrm{extension}}\mid G_{\mathrm{scaffold}},S)$",
    )
    fig.suptitle(
        r"Common $G_{\mathrm{small}}$ scaffold probability comparison (Supplementary)",
        y=1.02,
    )
    save(fig, figures, "common_gsmall_probability_methods")


def plot_structure(records: pd.DataFrame, figures: Path, prefix: str, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    violin_box(
        axes[0],
        records,
        f"{prefix}_structure_distance_normalized",
        "Normalized base-pair Hamming distance",
    )
    exact = [
        records.loc[
            records["method"].eq(method), f"{prefix}_structure_distance"
        ].eq(0).mean()
        for method in METHOD_ORDER
    ]
    axes[1].bar(
        np.arange(len(METHOD_ORDER)),
        exact,
        color=[COLORS[method] for method in METHOD_ORDER],
    )
    axes[1].set_xticks(
        np.arange(len(METHOD_ORDER)),
        [METHOD_LABELS[method] for method in METHOD_ORDER],
        rotation=18,
        ha="right",
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Exact-match rate")
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)
    fig.suptitle(title, y=1.02)
    save(fig, figures, f"common_gsmall_{prefix}_methods")


def build_assignment_sensitivity(
    current: pd.DataFrame, common_gbig: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in ("pkprobdesign", "desirna"):
        big = common_gbig.loc[common_gbig["method"].eq(method)].set_index("pkb_id")
        small = current.loc[current["method"].eq(method)].set_index("pkb_id")
        if set(big.index) != set(small.index):
            raise ValueError(f"{method}: common-G_big/common-G_small target coverage differs")
        for pkb_id in sorted(big.index):
            for assignment, source in (("big", big), ("small", small)):
                row = source.loc[pkb_id]
                rows.append(
                    {
                        "pkb_id": pkb_id,
                        "method": method,
                        "design_scaffold_assignment": assignment,
                        "evaluation_scaffold_assignment": assignment,
                        "sequence": row["sequence"],
                        "candidate_uid": row["candidate_uid"],
                        "logP_scaffold": row["logP_G"],
                        "logP_extension_given_scaffold": row[
                            "logP_G_prime_given_G_S"
                        ],
                        "log_pseudo_joint": row["combined_score"],
                        "scores_are_assignment_specific": True,
                        "same_full_target_probability_claimed": False,
                        "oracle_selection_performed": False,
                    }
                )
    return pd.DataFrame(rows)


def assignment_sensitivity_summary(source: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = (
        "logP_scaffold",
        "logP_extension_given_scaffold",
        "log_pseudo_joint",
    )
    for (method, assignment), group in source.groupby(
        ["method", "evaluation_scaffold_assignment"], sort=False
    ):
        row: dict[str, object] = {
            "method": method,
            "scaffold_assignment": assignment,
            "n_targets": group["pkb_id"].nunique(),
            "scores_are_assignment_specific": True,
            "same_full_target_probability_claimed": False,
            "oracle_selection_performed": False,
        }
        for metric in metrics:
            numeric = pd.to_numeric(group[metric], errors="coerce")
            finite = numeric.loc[np.isfinite(numeric)]
            row[f"{metric}_finite_n"] = len(finite)
            row[f"{metric}_mean_finite"] = finite.mean()
            row[f"{metric}_median_finite"] = finite.median()
        rows.append(row)
    return pd.DataFrame(rows)


def assignment_sensitivity_tests(source: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = (
        "logP_scaffold",
        "logP_extension_given_scaffold",
        "log_pseudo_joint",
    )
    for method in ("pkprobdesign", "desirna"):
        selected = source.loc[source["method"].eq(method)]
        for metric in metrics:
            pivot = selected.pivot(
                index="pkb_id",
                columns="evaluation_scaffold_assignment",
                values=metric,
            ).apply(pd.to_numeric, errors="coerce")
            finite_mask = np.isfinite(pivot[["big", "small"]]).all(axis=1)
            pair = pivot.loc[finite_mask, ["big", "small"]]
            advantage = pair["small"] - pair["big"]
            statistic, p_value, nonzero = wilcoxon_signed_rank(
                advantage, pd.Series(0.0, index=advantage.index)
            )
            tolerance = 1e-12
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "comparison": "common-G_small minus common-G_big",
                    "paired_targets_total": len(pivot),
                    "paired_targets_finite": len(pair),
                    "paired_targets_excluded_nonfinite": len(pivot) - len(pair),
                    "gsmall_higher": int((advantage > tolerance).sum()),
                    "gsmall_lower": int((advantage < -tolerance).sum()),
                    "ties": int((advantage.abs() <= tolerance).sum()),
                    "mean_gsmall_minus_gbig": advantage.mean(),
                    "median_gsmall_minus_gbig": advantage.median(),
                    "wilcoxon_nonzero_pairs": nonzero,
                    "wilcoxon_w_plus": statistic,
                    "p_value_raw": p_value,
                    "scores_are_assignment_specific": True,
                    "same_full_target_probability_claimed": False,
                    "oracle_selection_performed": False,
                }
            )
    frame = pd.DataFrame(rows)
    frame["bonferroni_within_method_family_size"] = frame.groupby("method")[
        "p_value_raw"
    ].transform(lambda values: int(values.notna().sum()))
    frame["p_value_bonferroni_within_method"] = frame.apply(
        lambda row: min(
            1.0,
            row["p_value_raw"] * row["bonferroni_within_method_family_size"],
        ),
        axis=1,
    )
    global_size = int(frame["p_value_raw"].notna().sum())
    frame["bonferroni_global_family_size"] = global_size
    frame["p_value_bonferroni_global"] = frame["p_value_raw"].map(
        lambda value: min(1.0, value * global_size)
    )
    return frame


def select_single_method_pool(frame: pd.DataFrame) -> pd.Series:
    errors = frame["probability_error"].fillna("").astype(str).str.strip()
    if errors.ne("").any():
        raise ValueError("DesiRNA cross-score pool contains evaluation errors")
    score = pd.to_numeric(frame["combined_score"], errors="coerce")
    if score.isna().any() or np.isposinf(score).any():
        raise ValueError("DesiRNA cross-score pool contains invalid probability values")
    ranked = frame.assign(_score=score, _rank=pd.to_numeric(frame["rank"], errors="raise"))
    ranked = ranked.sort_values(
        ["_score", "_rank", "sequence", "candidate_uid"],
        ascending=[False, True, True, True],
        kind="stable",
    )
    return ranked.iloc[0]


def read_cross_score(
    root: Path, target_ids: list[str], configuration: str, variant: str
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for pkb_id in target_ids:
        path = (
            root
            / "eval"
            / "desirna_cross_score"
            / "probability"
            / configuration
            / pkb_id
            / variant
            / "candidates.csv"
        )
        if not path.is_file():
            raise ValueError(f"missing DesiRNA cross-score table: {path}")
        frame = pd.read_csv(path, low_memory=False)
        if len(frame) != 20:
            raise ValueError(f"{pkb_id}/{configuration}: expected 20 rows")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_desirna_cross_score_table(
    current_probability: pd.DataFrame,
    common_gbig_probability: pd.DataFrame,
    root: Path,
    target_ids: list[str],
) -> pd.DataFrame:
    configurations: list[tuple[str, str, str, pd.DataFrame]] = [
        (
            "big",
            "big",
            "same_assignment_common_gbig",
            common_gbig_probability.loc[
                common_gbig_probability["method"].eq("desirna")
            ],
        ),
        (
            "small",
            "small",
            "same_assignment_common_gsmall",
            current_probability.loc[current_probability["method"].eq("desirna")],
        ),
        (
            "big",
            "small",
            "gbig_design_gsmall_eval",
            read_cross_score(root, target_ids, "gbig_design_gsmall_eval", "small"),
        ),
        (
            "small",
            "big",
            "gsmall_design_gbig_eval",
            read_cross_score(root, target_ids, "gsmall_design_gbig_eval", "big"),
        ),
    ]
    rows: list[dict[str, object]] = []
    for design, evaluation, configuration, frame in configurations:
        for pkb_id, pool in frame.groupby("pkb_id", sort=False):
            winner = select_single_method_pool(pool)
            rows.append(
                {
                    "pkb_id": pkb_id,
                    "method": "desirna",
                    "design_scaffold_assignment": design,
                    "evaluation_scaffold_assignment": evaluation,
                    "configuration": configuration,
                    "sequence": winner["sequence"],
                    "candidate_uid": winner["candidate_uid"],
                    "logP_scaffold": winner["logP_G"],
                    "logP_extension_given_scaffold": winner[
                        "logP_G_prime_given_G_S"
                    ],
                    "log_pseudo_joint": winner["combined_score"],
                    "pool_size": len(pool),
                    "selection_uses_target_distance": False,
                }
            )
    result = pd.DataFrame(rows)
    if len(result) != len(target_ids) * 4:
        raise ValueError("DesiRNA 2-by-2 cross-score table is incomplete")
    return result


def plot_assignment_sensitivity(source: pd.DataFrame, figures: Path) -> None:
    desirna = source.loc[source["method"].eq("desirna")]
    values: list[np.ndarray] = []
    for assignment in ("big", "small"):
        numeric = pd.to_numeric(
            desirna.loc[
                desirna["evaluation_scaffold_assignment"].eq(assignment),
                "log_pseudo_joint",
            ],
            errors="coerce",
        )
        values.append(numeric.loc[np.isfinite(numeric)].to_numpy(float))
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.violinplot(values, positions=(1, 2), showextrema=False)
    ax.boxplot(values, positions=(1, 2), widths=0.18, showfliers=False)
    ax.set_xticks(
        (1, 2),
        [r"$G_{\mathrm{big}}$ scaffold", r"$G_{\mathrm{small}}$ scaffold"],
    )
    ax.set_ylabel("log pseudo-joint probability")
    ax.set_title("DesiRNA scaffold-assignment sensitivity")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    save(fig, figures, "desirna_scaffold_assignment_sensitivity")


def main() -> int:
    args = parse_args()
    summary = args.results_root / "summary"
    figures = summary / "figures"
    summary.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    try:
        target_ids = read_target_ids(args.target_manifest)
        if len(target_ids) != args.expected_targets:
            raise ValueError(f"expected {args.expected_targets} targets")
        probability, nonfinite, evaluation_errors = classify_probability_pool(
            read_stage_tables(
                args.results_root,
                "probability",
                target_ids,
                expected_rows_per_target=len(METHOD_ORDER) * 20,
            ),
            args.expected_targets,
        )
        nonfinite.to_csv(summary / "nonfinite_probability_records.csv", index=False)
        evaluation_errors.to_csv(summary / "evaluation_errors.csv", index=False)
        if len(evaluation_errors):
            raise ValueError(
                f"probability pool contains {len(evaluation_errors)} implementation errors"
            )
        selected = validate_selected(
            read_stage_tables(args.results_root, "selected", target_ids),
            args.expected_targets,
        )
        validate_representative_selection(probability, selected)
        records = selected
        for stage in STAGES:
            records = merge_stage(
                records, read_stage_tables(args.results_root, stage, target_ids), stage
            )
        for error_column in ("two_stage_error", "viterbi_error", "ipknot_error"):
            errors = records[error_column].fillna("").astype(str).str.strip()
            if errors.ne("").any():
                raise ValueError(f"{error_column} contains {int(errors.ne('').sum())} errors")
        records = add_ipknot_distances(records)
        common_gbig = pd.read_csv(args.common_gbig_final_records, low_memory=False)
        common_gbig_probability = pd.read_csv(
            args.common_gbig_probability_records, low_memory=False
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    records["method_label"] = records["method"].map(METHOD_LABELS)
    records = records.sort_values(["pkb_id", "method"], kind="stable").reset_index(
        drop=True
    )
    probability.to_csv(summary / "all_probability_records.csv", index=False)
    selected.to_csv(summary / "selected_probability_representatives.csv", index=False)
    records.to_csv(summary / "final_records.csv", index=False)
    method_summary(records).to_csv(summary / "method_summary.csv", index=False)
    best = method_best_counts(records)
    best.to_csv(summary / "method_best_counts.csv", index=False)
    tests = paired_tests(records)
    tests.to_csv(summary / "paired_tests.csv", index=False)

    probability_source = records.loc[
        :,
        [
            "pkb_id",
            "method",
            "method_label",
            "logP_G",
            "logP_G_prime_given_G_S",
            "combined_score",
        ],
    ]
    probability_source.to_csv(
        summary / "common_gsmall_probability_figure_source.csv", index=False
    )
    probability_source.loc[
        probability_source["method"].isin(("pkprobdesign", "desirna"))
    ].to_csv(summary / "pkprobdesign_vs_desirna_probability_source.csv", index=False)
    tests.loc[
        tests["method_a"].eq("pkprobdesign") & tests["method_b"].eq("desirna")
    ].to_csv(summary / "pkprobdesign_vs_desirna_paired_tests.csv", index=False)

    for prefix in ("two_stage", "viterbi", "ipknot"):
        records.loc[
            :,
            [
                "pkb_id",
                "method",
                "method_label",
                f"{prefix}_structure_distance",
                f"{prefix}_structure_distance_normalized",
            ],
        ].to_csv(summary / f"{prefix}_figure_source.csv", index=False)

    sensitivity = build_assignment_sensitivity(records, common_gbig)
    sensitivity.to_csv(summary / "scaffold_assignment_sensitivity.csv", index=False)
    cross_score = build_desirna_cross_score_table(
        probability,
        common_gbig_probability,
        args.results_root,
        target_ids,
    )
    cross_score.to_csv(summary / "desirna_scaffold_cross_score_2x2.csv", index=False)
    assignment_sensitivity_summary(sensitivity).to_csv(
        summary / "scaffold_assignment_sensitivity_summary.csv", index=False
    )
    assignment_sensitivity_tests(sensitivity).to_csv(
        summary / "scaffold_assignment_sensitivity_paired_tests.csv", index=False
    )
    cross_score.groupby(
        ["design_scaffold_assignment", "evaluation_scaffold_assignment"],
        as_index=False,
    ).agg(
        n_targets=("pkb_id", "nunique"),
        mean_logP_scaffold=("logP_scaffold", "mean"),
        median_logP_scaffold=("logP_scaffold", "median"),
        mean_logP_extension_given_scaffold=(
            "logP_extension_given_scaffold",
            "mean",
        ),
        median_logP_extension_given_scaffold=(
            "logP_extension_given_scaffold",
            "median",
        ),
        mean_log_pseudo_joint=("log_pseudo_joint", "mean"),
        median_log_pseudo_joint=("log_pseudo_joint", "median"),
    ).to_csv(summary / "desirna_scaffold_cross_score_2x2_summary.csv", index=False)

    plot_probability(records, figures)
    plot_structure(
        records,
        figures,
        "two_stage",
        r"Common $G_{\mathrm{small}}$: two-stage LinearFold $\rightarrow$ SCFG2 Viterbi",
    )
    plot_structure(
        records,
        figures,
        "viterbi",
        r"Common $G_{\mathrm{small}}$: HotKnots-guided SCFG2 Viterbi",
    )
    plot_structure(
        records,
        figures,
        "ipknot",
        r"Common $G_{\mathrm{small}}$: IPknot prediction (Supplementary)",
    )
    plot_assignment_sensitivity(sensitivity, figures)

    manifest = {
        "workflow": "common_gsmall_four_method_supplementary_aggregation",
        "analysis_role": "Supplementary scaffold-assignment sensitivity; common-G_big remains primary",
        "target_count": args.expected_targets,
        "method_count": len(METHOD_ORDER),
        "final_record_count": len(records),
        "candidate_probability_record_count": len(probability),
        "candidate_uid_unique_count": probability["candidate_uid"].nunique(),
        "target_method_rank_unique_count": len(
            probability.loc[:, ["pkb_id", "method", "rank"]].drop_duplicates()
        ),
        "genuine_nonfinite_probability_term_count": len(nonfinite),
        "evaluation_error_count": len(evaluation_errors),
        "representative_selection_reproduced": True,
        "selection_rule": "maximum common-G_small pseudo-joint probability within each 20-candidate pool",
        "selection_uses_target_distance": False,
        "oracle_assignment_selection_performed": False,
        "scores_are_assignment_specific_not_identical_full_target_probabilities": True,
        "target_manifest": str(args.target_manifest.resolve()),
        "common_gbig_final_records": str(args.common_gbig_final_records.resolve()),
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent.parent), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "aggregation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (summary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} final common-G_small records to {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
