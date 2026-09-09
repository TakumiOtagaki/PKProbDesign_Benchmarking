#!/usr/bin/env python3
"""Normalize one scaffold-oriented DesiRNA result into exactly 20 candidates."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from evaluation_utils import dotbracket_to_pairs, normalize_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkb-id", required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--raw-results-csv", type=Path, required=True)
    parser.add_argument("--runtime-sec", type=float, required=True)
    parser.add_argument("--command-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    parser.add_argument("--repo-revision", required=True)
    parser.add_argument("--desirna-revision", required=True)
    parser.add_argument("--viennarna-version", required=True)
    parser.add_argument("--expected-candidates", type=int, default=20)
    parser.add_argument("--time-budget-sec", type=int, default=60)
    parser.add_argument("--replicas", type=int, default=10)
    parser.add_argument("--exchange-interval", type=int, default=100)
    parser.add_argument("--turner-parameters", default="1999")
    parser.add_argument("--score", default="Ed-Epf:1.0")
    parser.add_argument("--seed-number", type=int, default=0)
    parser.add_argument("--design-target-column", default="recolored_target")
    parser.add_argument("--scaffold-column", default="g_big_dotbracket")
    parser.add_argument("--extension-column", default="g_small_dotbracket")
    parser.add_argument("--scaffold-variant", choices=("big", "small"), default="big")
    parser.add_argument("--extension-family", default="[]")
    parser.add_argument("--decomposition-source", default="crossing_graph_maximum_g_big")
    parser.add_argument("--workflow", default="common_gbig_desirna_rerun")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_target(path: Path, pkb_id: str) -> dict[str, str]:
    matches = [row for row in read_csv(path) if row.get("pkb_id") == pkb_id]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest row for {pkb_id}, found {len(matches)}")
    return matches[0]


def normalize_rows(
    *,
    pkb_id: str,
    target: dict[str, str],
    source_rows: list[dict[str, str]],
    source_path: Path,
    runtime_sec: float,
    expected_candidates: int,
    design_target_column: str = "recolored_target",
    scaffold_column: str = "g_big_dotbracket",
    extension_column: str = "g_small_dotbracket",
    scaffold_variant: str = "big",
    extension_family: str = "[]",
    decomposition_source: str = "crossing_graph_maximum_g_big",
) -> list[dict[str, object]]:
    if len(source_rows) < expected_candidates:
        raise ValueError(
            f"{pkb_id}: expected at least {expected_candidates} DesiRNA rows, "
            f"found {len(source_rows)}"
        )
    length = int(target["length"])
    for column in (design_target_column, scaffold_column, extension_column):
        if column not in target:
            raise ValueError(f"{pkb_id}: target manifest is missing {column!r}")
    design_target = target[design_target_column]
    original = target["original_target"]
    if dotbracket_to_pairs(design_target) != dotbracket_to_pairs(original):
        raise ValueError(f"{pkb_id}: target manifest does not preserve the pair map")

    output: list[dict[str, object]] = []
    for rank, source in enumerate(source_rows[:expected_candidates], start=1):
        sequence = normalize_sequence(source.get("sequence") or "")
        prediction = (source.get("mfe_ss") or "").strip()
        if len(sequence) != length or set(sequence) - set("ACGU"):
            raise ValueError(f"{pkb_id}/rank={rank}: invalid designed sequence")
        if len(prediction) != length:
            raise ValueError(f"{pkb_id}/rank={rank}: predicted structure length mismatch")
        dotbracket_to_pairs(prediction)
        try:
            epf = float(source.get("Epf") or "")
            score = float(source.get("scoring_function") or "")
            mcc = float(source.get("mcc") or "")
        except ValueError as error:
            raise ValueError(f"{pkb_id}/rank={rank}: nonnumeric score field") from error
        if not all(math.isfinite(value) for value in (epf, score, mcc, runtime_sec)):
            raise ValueError(f"{pkb_id}/rank={rank}: nonfinite score/runtime field")
        output.append(
            {
                "pkb_id": pkb_id,
                "method": "desirna",
                "rank": rank,
                "sequence": sequence,
                "target_dotbracket": design_target,
                "original_target_dotbracket": original,
                "target_g_dotbracket": target[scaffold_column],
                "target_gprime_dotbracket": target[extension_column],
                "target_gprime_family": extension_family,
                "scaffold_variant": scaffold_variant,
                "decomposition_source": decomposition_source,
                "length": length,
                "phase1_predicted_structure": prediction,
                "phase1_objective": score,
                "partition_function_energy": epf,
                "one_mcc": mcc,
                "runtime_sec": runtime_sec,
                "raw_source_path": str(source_path.resolve()),
                "target_manifest_path": str(target.get("target_manifest_path") or ""),
                "raw_record_json": json.dumps(source, sort_keys=True),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    try:
        target = load_target(args.target_manifest, args.pkb_id)
        target["target_manifest_path"] = str(args.target_manifest.resolve())
        source_rows = read_csv(args.raw_results_csv)
        rows = normalize_rows(
            pkb_id=args.pkb_id,
            target=target,
            source_rows=source_rows,
            source_path=args.raw_results_csv,
            runtime_sec=args.runtime_sec,
            expected_candidates=args.expected_candidates,
            design_target_column=args.design_target_column,
            scaffold_column=args.scaffold_column,
            extension_column=args.extension_column,
            scaffold_variant=args.scaffold_variant,
            extension_family=args.extension_family,
            decomposition_source=args.decomposition_source,
        )
        write_csv(args.output, rows)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    provenance = {
        "workflow": args.workflow,
        "design_target_column": args.design_target_column,
        "scaffold_column": args.scaffold_column,
        "extension_column": args.extension_column,
        "scaffold_variant": args.scaffold_variant,
        "decomposition_source": args.decomposition_source,
        "pkb_id": args.pkb_id,
        "target_manifest": str(args.target_manifest.resolve()),
        "raw_results_csv": str(args.raw_results_csv.resolve()),
        "normalized_candidates": str(args.output.resolve()),
        "candidate_count": len(rows),
        "runtime_sec": args.runtime_sec,
        "command_file": str(args.command_file.resolve()),
        "command": args.command_file.read_text(encoding="utf-8").strip(),
        "repo_revision": args.repo_revision,
        "desirna_revision": args.desirna_revision,
        "viennarna_version": args.viennarna_version,
        "settings": {
            "score": args.score,
            "pseudoknot_mode": "enabled_by_pseudoknotted_target",
            "turner_parameters": args.turner_parameters,
            "replicas": args.replicas,
            "exchange_interval": args.exchange_interval,
            "time_budget_sec": args.time_budget_sec,
            "returned_designs": args.expected_candidates,
            "acgu_content_control": "off",
            "negative_design": "off",
            "seed_number": args.seed_number,
            "seed_policy": (
                "DesiRNA default stochastic seed (seed_number=0)"
                if args.seed_number == 0
                else "DesiRNA explicit deterministic seed number"
            ),
        },
    }
    args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
    args.provenance_output.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Normalized {len(rows)} DesiRNA candidates for {args.pkb_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
