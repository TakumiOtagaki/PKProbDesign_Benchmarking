#!/usr/bin/env python3
"""Aggregate the final four-method common-G_big comparison and figures."""
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

from compute_benchmark_pvalues import wilcoxon_signed_rank
from evaluation_utils import base_pair_distance
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
STAGES = ("two_stage_mfe", "hotknots_viterbi", "ipknot_supplementary")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--old-comparison-root", type=Path, default=None)
    parser.add_argument("--expected-targets", type=int, default=254)
    return parser.parse_args()


def read_target_ids(path: Path) -> list[str]:
    frame = pd.read_csv(path, usecols=["pkb_id"])
    ids = frame["pkb_id"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("target manifest contains duplicate IDs")
    return ids


def read_stage_tables(
    root: Path, stage: str, target_ids: list[str], expected_rows_per_target: int = 4
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for pkb_id in target_ids:
        path = root / "eval" / stage / pkb_id / "big" / "candidates.csv"
        if not path.is_file():
            raise ValueError(f"missing {stage} table: {path}")
        frame = pd.read_csv(path, low_memory=False)
        if len(frame) != expected_rows_per_target:
            raise ValueError(
                f"{pkb_id}/{stage}: expected {expected_rows_per_target} rows, "
                f"found {len(frame)}"
            )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def validate_probability_pool(frame: pd.DataFrame, expected_targets: int) -> pd.DataFrame:
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
    for column in ("logP_G", "logP_G_prime_given_G_S", "combined_score"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"probability pool {column} contains nonfinite values")
        if frame[column].gt(1e-8).any():
            raise ValueError(f"probability pool {column} contains values greater than zero")
    errors = frame["probability_error"].fillna("").astype(str).str.strip()
    if errors.ne("").any():
        raise ValueError("probability pool contains evaluation errors")
    return frame


def validate_representative_selection(
    probability: pd.DataFrame, selected: pd.DataFrame
) -> None:
    recomputed_uids: set[str] = set()
    for _, target in probability.groupby("pkb_id", sort=False):
        recomputed = select_representatives(
            target.fillna("").astype(str).to_dict("records"), per_method=20
        )
        recomputed_uids.update(row["candidate_uid"] for row in recomputed)
    if recomputed_uids != set(selected["candidate_uid"]):
        raise ValueError("selected representatives do not reproduce from probability-only rule")


def validate_selected(frame: pd.DataFrame, expected_targets: int) -> pd.DataFrame:
    if frame["pkb_id"].nunique() != expected_targets:
        raise ValueError("selected table target count mismatch")
    counts = frame.groupby(["pkb_id", "method"]).size()
    if set(frame["method"]) != set(METHOD_ORDER) or not counts.eq(1).all():
        raise ValueError("selected table must contain one row per target and method")
    if frame["candidate_uid"].duplicated().any():
        raise ValueError("selected candidate UIDs are not unique")
    if not frame["selection_uses_target_distance"].astype(str).str.lower().eq("false").all():
        raise ValueError("representative selection was not declared distance-independent")
    for column in ("logP_G", "logP_G_prime_given_G_S", "combined_score"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"{column} contains nonfinite values")
        if frame[column].gt(1e-8).any():
            raise ValueError(f"{column} contains values greater than zero")
    errors = frame["probability_error"].fillna("").astype(str).str.strip()
    if errors.ne("").any():
        raise ValueError("selected rows contain probability errors")
    return frame


def merge_stage(selected: pd.DataFrame, stage: pd.DataFrame, stage_name: str) -> pd.DataFrame:
    if stage["candidate_uid"].duplicated().any():
        raise ValueError(f"{stage_name}: duplicate candidate UIDs")
    if set(stage["candidate_uid"]) != set(selected["candidate_uid"]):
        raise ValueError(f"{stage_name}: representative coverage mismatch")
    extra = [column for column in stage if column not in selected.columns]
    merged = selected.merge(
        stage.loc[:, ["candidate_uid", *extra]],
        on="candidate_uid",
        how="left",
        validate="one_to_one",
    )
    return merged


def add_ipknot_distances(records: pd.DataFrame) -> pd.DataFrame:
    distances: list[int] = []
    normalized: list[float] = []
    for row in records.itertuples(index=False):
        distance = base_pair_distance(row.ipknot_predicted_structure, row.target_dotbracket)
        distances.append(distance)
        normalized.append(distance / int(row.length))
    records["ipknot_structure_distance"] = distances
    records["ipknot_structure_distance_normalized"] = normalized
    return records


def method_summary(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        selected = records.loc[records["method"].eq(method)]
        row: dict[str, object] = {
            "method": method,
            "method_label": METHOD_LABELS[method],
            "n_targets": selected["pkb_id"].nunique(),
            "mean_log_pseudo_joint": selected["combined_score"].mean(),
            "median_log_pseudo_joint": selected["combined_score"].median(),
            "mean_logP_G": selected["logP_G"].mean(),
            "median_logP_G": selected["logP_G"].median(),
            "mean_logP_Gprime_given_G_S": selected["logP_G_prime_given_G_S"].mean(),
            "median_logP_Gprime_given_G_S": selected["logP_G_prime_given_G_S"].median(),
        }
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


def method_best_counts(records: pd.DataFrame) -> pd.DataFrame:
    counters = {
        method: {"unique_best_count": 0, "tied_best_count": 0, "best_including_ties": 0}
        for method in METHOD_ORDER
    }
    targets_with_ties = 0
    for _, target in records.groupby("pkb_id", sort=False):
        best = target["combined_score"].max()
        winners = target.loc[np.isclose(target["combined_score"], best, rtol=1e-12, atol=1e-12)]
        tied = len(winners) > 1
        targets_with_ties += tied
        for method in winners["method"]:
            counters[method]["best_including_ties"] += 1
            key = "tied_best_count" if tied else "unique_best_count"
            counters[method][key] += 1
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
            pair = pivot[[method_a, method_b]].dropna()
            advantage = pair[method_a] - pair[method_b]
            if not higher_is_better:
                advantage = -advantage
            statistic, p_value, nonzero = wilcoxon_signed_rank(
                advantage, pd.Series(0.0, index=advantage.index)
            )
            tolerance = 1e-12
            rows.append(
                {
                    "metric": metric,
                    "higher_is_better": higher_is_better,
                    "method_a": method_a,
                    "method_b": method_b,
                    "paired_targets": len(pair),
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
    valid_tests = int(frame["p_value_raw"].notna().sum())
    frame["bonferroni_global_family_size"] = valid_tests
    frame["p_value_bonferroni_global"] = frame["p_value_raw"].map(
        lambda value: min(1.0, value * valid_tests) if pd.notna(value) else np.nan
    )
    frame["bonferroni_metric_family_size"] = frame.groupby("metric")[
        "p_value_raw"
    ].transform(lambda series: int(series.notna().sum()))
    frame["p_value_bonferroni"] = frame.apply(
        lambda row: (
            min(1.0, row["p_value_raw"] * row["bonferroni_metric_family_size"])
            if pd.notna(row["p_value_raw"])
            else np.nan
        ),
        axis=1,
    )
    return frame


def violin_box(ax: plt.Axes, records: pd.DataFrame, column: str, ylabel: str) -> None:
    values = [records.loc[records["method"].eq(method), column].to_numpy(float) for method in METHOD_ORDER]
    positions = np.arange(1, len(METHOD_ORDER) + 1)
    violin = ax.violinplot(values, positions=positions, showextrema=False)
    for body, method in zip(violin["bodies"], METHOD_ORDER):
        body.set_facecolor(COLORS[method])
        body.set_edgecolor(COLORS[method])
        body.set_alpha(0.35)
    boxes = ax.boxplot(values, positions=positions, widths=0.18, showfliers=False, patch_artist=True)
    for box, method in zip(boxes["boxes"], METHOD_ORDER):
        box.set_facecolor("white")
        box.set_edgecolor(COLORS[method])
    ax.set_xticks(positions, [METHOD_LABELS[m] for m in METHOD_ORDER], rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.3)


def save(fig: plt.Figure, figures: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(figures / f"{stem}.png", dpi=250, bbox_inches="tight")
    fig.savefig(figures / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_probability(records: pd.DataFrame, figures: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    violin_box(axes[0], records, "combined_score", "log pseudo-joint probability")
    violin_box(axes[1], records, "logP_G", "log P(G_big | S)")
    violin_box(
        axes[2], records, "logP_G_prime_given_G_S", "log P(G_small | G_big, S)"
    )
    fig.suptitle("Common-G_big probability comparison", y=1.02)
    save(fig, figures, "common_gbig_probability_methods")


def plot_structure(records: pd.DataFrame, figures: Path, prefix: str, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    violin_box(
        axes[0],
        records,
        f"{prefix}_structure_distance_normalized",
        "Normalized base-pair Hamming distance",
    )
    exact = [
        records.loc[records["method"].eq(method), f"{prefix}_structure_distance"].eq(0).mean()
        for method in METHOD_ORDER
    ]
    axes[1].bar(
        np.arange(len(METHOD_ORDER)), exact, color=[COLORS[method] for method in METHOD_ORDER]
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
    save(fig, figures, f"common_gbig_{prefix}_methods")


def case_study(
    records: pd.DataFrame, summary: Path, figures: Path, old_comparison_root: Path | None
) -> dict[str, object]:
    selected = records.loc[
        records["pkb_id"].eq("PKB00076") & records["method"].eq("desirna")
    ].copy()
    if len(selected) != 1:
        raise ValueError("PKB00076 DesiRNA representative is missing or duplicated")
    selected.to_csv(summary / "pkb00076_desirna_case_source.csv", index=False)
    row = selected.iloc[0]
    old_sequence = ""
    if old_comparison_root is not None:
        old_path = old_comparison_root / "summary" / "five_way_records.csv"
        old = pd.read_csv(old_path, low_memory=False)
        matches = old.loc[old["pkb_id"].eq("PKB00076") & old["method"].eq("desirna")]
        if len(matches) != 1:
            raise ValueError("old PKB00076 DesiRNA representative lookup failed")
        old_sequence = str(matches.iloc[0]["sequence"])
    report = {
        "pkb_id": "PKB00076",
        "old_sequence": old_sequence,
        "new_sequence": str(row["sequence"]),
        "representative_changed": bool(old_sequence and old_sequence != row["sequence"]),
        "old_comparison_root": "" if old_comparison_root is None else str(old_comparison_root.resolve()),
    }
    (summary / "pkb00076_desirna_case_comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        ("Target", str(row["target_dotbracket"])),
        ("G_big", str(row["target_g_dotbracket"])),
        ("G_small", str(row["target_gprime_dotbracket"])),
        ("Sequence", str(row["sequence"])),
        ("Two-stage", str(row["two_stage_predicted_structure"])),
        ("HotKnots-guided", str(row["viterbi_predicted_structure"])),
        ("IPknot", str(row["ipknot_predicted_structure"])),
    ]
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.axis("off")
    ax.set_title("PKB00076 — DesiRNA representative under common G_big", loc="left")
    y = 0.88
    for label, value in lines:
        ax.text(0.01, y, f"{label:16s}", fontfamily="monospace", fontweight="bold", va="top")
        ax.text(0.18, y, value, fontfamily="monospace", va="top")
        y -= 0.105
    ax.text(
        0.01,
        0.07,
        f"log pseudo-joint = {row['combined_score']:.3f}; representative changed = {report['representative_changed']}",
    )
    save(fig, figures, "pkb00076_desirna_common_gbig")
    return report


def main() -> int:
    args = parse_args()
    try:
        target_ids = read_target_ids(args.target_manifest)
        if len(target_ids) != args.expected_targets:
            raise ValueError(f"expected {args.expected_targets} targets")
        probability = validate_probability_pool(
            read_stage_tables(
                args.results_root,
                "probability",
                target_ids,
                expected_rows_per_target=len(METHOD_ORDER) * 20,
            ),
            args.expected_targets,
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
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    records["method_label"] = records["method"].map(METHOD_LABELS)
    records = records.sort_values(
        ["pkb_id", "method"],
        key=lambda series: (
            series.map({method: i for i, method in enumerate(METHOD_ORDER)})
            if series.name == "method"
            else series
        ),
        kind="stable",
    ).reset_index(drop=True)
    summary = args.results_root / "summary"
    figures = summary / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    probability.loc[
        :,
        [
            "pkb_id",
            "method",
            "rank",
            "sequence",
            "candidate_uid",
            "length",
            "target_dotbracket",
            "target_g_dotbracket",
            "target_gprime_dotbracket",
            "logP_G",
            "logP_G_prime_given_G_S",
            "combined_score",
            "probability_error",
        ],
    ].to_csv(summary / "all_probability_records.csv", index=False)
    selected.loc[
        :,
        [
            "pkb_id",
            "method",
            "rank",
            "sequence",
            "candidate_uid",
            "length",
            "target_dotbracket",
            "target_g_dotbracket",
            "target_gprime_dotbracket",
            "logP_G",
            "logP_G_prime_given_G_S",
            "combined_score",
            "representative_selection_rule",
            "selection_uses_target_distance",
        ],
    ].to_csv(summary / "selected_probability_representatives.csv", index=False)
    records.to_csv(summary / "final_records.csv", index=False)
    method_summary(records).to_csv(summary / "method_summary.csv", index=False)
    method_best_counts(records).to_csv(summary / "method_best_counts.csv", index=False)
    paired_tests(records).to_csv(summary / "paired_tests.csv", index=False)
    probability_source = records.loc[
        :, ["pkb_id", "method", "method_label", "logP_G", "logP_G_prime_given_G_S", "combined_score"]
    ]
    probability_source.to_csv(summary / "common_gbig_probability_figure_source.csv", index=False)
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
    plot_probability(records, figures)
    plot_structure(records, figures, "two_stage", "Two-stage LinearFold → SCFG2 Viterbi")
    plot_structure(records, figures, "viterbi", "HotKnots-guided SCFG2 Viterbi")
    plot_structure(records, figures, "ipknot", "IPknot prediction (Supplementary)")
    case_report = case_study(records, summary, figures, args.old_comparison_root)
    manifest = {
        "workflow": "common_gbig_four_method_final_aggregation",
        "target_count": args.expected_targets,
        "method_count": len(METHOD_ORDER),
        "final_record_count": len(records),
        "methods": list(METHOD_ORDER),
        "candidate_probability_record_count": len(probability),
        "candidate_probability_finite_count": int(
            np.isfinite(
                probability[["logP_G", "logP_G_prime_given_G_S", "combined_score"]]
            ).sum().sum()
        ),
        "expected_candidate_probability_finite_count": len(probability) * 3,
        "representative_probability_finite_count": int(
            np.isfinite(
                records[["logP_G", "logP_G_prime_given_G_S", "combined_score"]]
            ).sum().sum()
        ),
        "representative_selection_reproduced": True,
        "evaluation_error_count": 0,
        "selection_rule": "maximum common-G_big pseudo-joint probability within each 20-candidate pool",
        "selection_uses_target_distance": False,
        "target_manifest": str(args.target_manifest.resolve()),
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent.parent), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "aggregation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pkb00076_representative_changed": case_report["representative_changed"],
    }
    (summary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(records)} final common-G_big records to {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
