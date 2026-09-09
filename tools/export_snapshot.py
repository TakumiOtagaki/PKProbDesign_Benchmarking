#!/usr/bin/env python3
"""Maintainer-only, allowlisted export from the original benchmark checkout.

No Git metadata, raw logs, third-party software or natural RNA sequences are
copied. CSV values are retained verbatim; only columns and filenames change.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

POOL_FIELDS = (
    "pkb_id method rank sequence candidate_uid length target_dotbracket "
    "target_g_dotbracket target_gprime_dotbracket logP_G "
    "logP_G_prime_given_G_S combined_score probability_error"
).split()
SELECTED_FIELDS = POOL_FIELDS + (
    "method_label representative_selection_rule selection_uses_target_distance "
    "two_stage_seed_structure_union two_stage_seed_g_dotbracket "
    "two_stage_seed_gprime_dotbracket two_stage_predicted_structure "
    "two_stage_generated_structure two_stage_energy two_stage_error "
    "two_stage_structure_distance two_stage_structure_distance_normalized "
    "viterbi_predicted_structure viterbi_generated_structure viterbi_energy "
    "viterbi_error viterbi_scaffold_count viterbi_scaffold_index "
    "viterbi_scaffold_structure viterbi_scaffold_energy viterbi_scaffold_source "
    "hotknots_threshold hotknots_topk hotknots_empty_included hotknots_rank "
    "hotknots_alignment_score viterbi_structure_distance "
    "viterbi_structure_distance_normalized ipknot_predicted_structure "
    "ipknot_model ipknot_thresholds ipknot_error ipknot_structure_distance "
    "ipknot_structure_distance_normalized"
).split()
TARGET_FIELDS = (
    "pkb_id length original_target recolored_target g_big_dotbracket "
    "g_small_dotbracket target_pair_count g_big_pair_count g_small_pair_count"
).split()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Refusing to overwrite a nonempty export directory")
    bundle = source / "results/strict_density2_big_small/runs/dp09_vpborderfix_adamfix_dedup254"
    provenance: dict[str, object] = {
        "description": "Column-only export; no sequence design or probability recomputation",
        "excluded": ["git history", "natural sequences", "raw logs", "absolute paths", "third-party code and binaries"],
        "files": {},
    }

    def export(original: Path, destination: str, fields: list[str] | None = None) -> None:
        rows = read(original)
        fields = fields or list(rows[0])
        path = output / destination
        write(path, rows, fields)
        provenance["files"][destination] = {
            "source": str(original.relative_to(source)),
            "source_sha256": sha256(original),
            "export_sha256": sha256(path),
            "rows": len(rows),
            "columns": fields,
            "values_unchanged": True,
        }

    for assignment in ("big", "small"):
        summary = bundle / f"common_g{assignment}_desirna_rerun_20260805/evaluation/summary"
        export(summary / "all_probability_records.csv", f"common_g{assignment}/candidate_pool.csv", POOL_FIELDS)
        export(summary / "final_records.csv", f"common_g{assignment}/representatives.csv", SELECTED_FIELDS)
        for name in ("method_summary", "method_best_counts", "paired_tests"):
            export(summary / f"{name}.csv", f"common_g{assignment}/reference/{name}.csv")

    target_source = bundle / "common_gbig_desirna_rerun_20260805/targets/recolored_targets.csv"
    targets = read(target_source)
    joblist = source / "pseudobasepp/dataset_seqlen100_strict_density2_minloop3_dedup254_joblist.txt"
    ids = [line.split("|", 1)[0] for line in joblist.read_text().splitlines() if line.strip()]
    if len(ids) != 254 or len(set(ids)) != 254 or set(ids) != {row["pkb_id"] for row in targets}:
        raise SystemExit("Target identity/count mismatch")
    index = {pkb_id: str(i) for i, pkb_id in enumerate(ids, 1)}
    for row in targets:
        row["target_index"] = index[row["pkb_id"]]
    targets.sort(key=lambda row: int(row["target_index"]))
    write(output / "targets.csv", targets, ["target_index", *TARGET_FIELDS])
    provenance["files"]["targets.csv"] = {
        "source": str(target_source.relative_to(source)),
        "source_sha256": sha256(target_source),
        "source_joblist_sha256": sha256(joblist),
        "export_sha256": sha256(output / "targets.csv"),
        "rows": len(targets),
        "columns": ["target_index", *TARGET_FIELDS],
        "transformation": "Project structure columns; add 1-based index from original deduplicated joblist; omit natural sequences",
    }
    (output / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Exported {len(provenance['files'])} tables to {output}")


if __name__ == "__main__":
    main()
