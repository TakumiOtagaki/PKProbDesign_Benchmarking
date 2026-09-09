#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import os
from pathlib import Path
import subprocess
import time

from evaluation_utils import (
    base_pair_distance,
    normalized_distance,
    sequence_length,
    split_target_dotbracket,
)
from evaluate_candidate_viterbi import read_rows, write_rows


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_PREDICTOR = REPO_ROOT / "bin" / "scfg2_hotspot_predictor"
DEFAULT_WRAPPER = REPO_ROOT / "bin" / "cparty_hotknots_hotspot_wrapper"
DEFAULT_HOTKNOTS_ROOT = REPO_ROOT / "build" / "hotknots-v2"
DEFAULT_PARAM_FILE = REPO_ROOT / "submodules" / "CParty" / "params" / "rna_DirksPierce09.par"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SCFG2 Viterbi predictions over HotKnots v2 stem-seed scaffolds."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--predictor-bin",
        type=Path,
        default=Path(os.environ.get("CPARTY_HOTSPOT_PREDICTOR_BIN", str(DEFAULT_PREDICTOR))),
    )
    parser.add_argument(
        "--wrapper-bin",
        type=Path,
        default=Path(os.environ.get("CPARTY_HOTKNOTS_WRAPPER", str(DEFAULT_WRAPPER))),
    )
    parser.add_argument(
        "--hotknots-root",
        type=Path,
        default=Path(os.environ.get("CPARTY_HOTKNOTS_ROOT", str(DEFAULT_HOTKNOTS_ROOT))),
    )
    parser.add_argument(
        "--param-file",
        type=Path,
        default=Path(os.environ.get("CPARTY_VITERBI_PARAM_FILE", str(DEFAULT_PARAM_FILE))),
    )
    parser.add_argument("--hotspot-topk", type=int, default=20)
    parser.add_argument("--hotspot-threshold", type=int, default=400)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--method",
        dest="methods",
        action="append",
        default=[],
        help="Evaluate only this candidate method (repeatable).",
    )
    return parser.parse_args()


def filter_rows_by_methods(
    rows: list[dict[str, str]], methods: list[str]
) -> list[dict[str, str]]:
    requested = {method.strip().lower() for method in methods if method.strip()}
    if not requested:
        return rows
    selected = [
        row
        for row in rows
        if (row.get("method") or "").strip().lower() in requested
    ]
    if not selected:
        raise ValueError(
            "no candidate rows matched --method: " + ", ".join(sorted(requested))
        )
    return selected


def parse_predictor_output(text: str) -> tuple[list[dict[str, str]], dict[str, str]]:
    rows = list(csv.DictReader(text.splitlines(), delimiter="\t"))
    if not rows:
        raise RuntimeError("hotspot predictor output contained no candidates")
    selected = [row for row in rows if (row.get("selected") or "").strip() == "1"]
    if len(selected) != 1:
        raise RuntimeError(f"hotspot predictor selected {len(selected)} candidates instead of one")
    return rows, selected[0]


def run_predictor(
    sequence: str,
    *,
    predictor_bin: Path,
    wrapper_bin: Path,
    hotknots_root: Path,
    param_file: Path,
    hotspot_topk: int,
    hotspot_threshold: int,
) -> dict[str, str]:
    command = [
        str(predictor_bin),
        "--sequence",
        sequence,
        "--wrapper",
        str(wrapper_bin),
        "--hotknots-root",
        str(hotknots_root),
        "--max-candidates",
        str(hotspot_topk),
        "--threshold",
        str(hotspot_threshold),
    ]
    if param_file:
        command.extend(["--param-file", str(param_file)])
    start = time.monotonic()
    completed = subprocess.run(command, capture_output=True, text=True)
    elapsed = time.monotonic() - start
    base = {
        "viterbi_runtime_sec": f"{elapsed:.6f}",
        "viterbi_scaffold_mode": "hotknots_v2_with_empty",
        "hotknots_threshold": str(hotspot_threshold),
        "hotknots_topk": str(hotspot_topk),
        "hotknots_empty_included": "true",
    }
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "hotspot predictor failed"
        return {
            **base,
            "viterbi_predicted_structure": "",
            "viterbi_generated_structure": "",
            "viterbi_energy": "",
            "viterbi_error": detail,
            "viterbi_scaffold_count": "",
            "viterbi_scaffold_index": "",
            "viterbi_scaffold_structure": "",
            "viterbi_scaffold_energy": "",
            "viterbi_scaffold_source": "",
            "hotknots_rank": "",
            "hotknots_alignment_score": "",
        }
    try:
        candidates, selected = parse_predictor_output(completed.stdout)
    except Exception as error:
        return {
            **base,
            "viterbi_predicted_structure": "",
            "viterbi_generated_structure": "",
            "viterbi_energy": "",
            "viterbi_error": str(error),
            "viterbi_scaffold_count": "",
            "viterbi_scaffold_index": "",
            "viterbi_scaffold_structure": "",
            "viterbi_scaffold_energy": "",
            "viterbi_scaffold_source": "",
            "hotknots_rank": "",
            "hotknots_alignment_score": "",
        }
    return {
        **base,
        "viterbi_predicted_structure": selected.get("union_structure", ""),
        "viterbi_generated_structure": selected.get("generated_structure", ""),
        "viterbi_energy": selected.get("energy", ""),
        "viterbi_error": selected.get("error", ""),
        "viterbi_scaffold_count": str(len(candidates)),
        "viterbi_scaffold_index": selected.get("candidate_index", ""),
        "viterbi_scaffold_structure": selected.get("scaffold_structure", ""),
        # The old column is retained for schema compatibility, but the
        # HotKnots alignment score is deliberately not mislabeled as energy.
        "viterbi_scaffold_energy": "",
        "viterbi_scaffold_source": selected.get("source", ""),
        "hotknots_rank": selected.get("hotknots_rank", ""),
        "hotknots_alignment_score": selected.get("alignment_score", ""),
    }


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if not 1 <= args.hotspot_topk <= 20:
        raise SystemExit("--hotspot-topk must be in 1..20")
    if args.hotspot_threshold <= 0:
        raise SystemExit("--hotspot-threshold must be positive")
    rows = read_rows(args.input_csv)
    if not rows:
        raise SystemExit(f"No rows found in {args.input_csv}")
    try:
        rows = filter_rows_by_methods(rows, args.methods)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    sequences: list[str] = []
    seen: set[str] = set()
    for row in rows:
        sequence = (row.get("sequence") or "").strip().upper().replace("T", "U")
        if sequence and sequence not in seen:
            seen.add(sequence)
            sequences.append(sequence)

    def evaluate(sequence: str) -> tuple[str, dict[str, str]]:
        return sequence, run_predictor(
            sequence,
            predictor_bin=args.predictor_bin,
            wrapper_bin=args.wrapper_bin,
            hotknots_root=args.hotknots_root,
            param_file=args.param_file,
            hotspot_topk=args.hotspot_topk,
            hotspot_threshold=args.hotspot_threshold,
        )

    cache: dict[str, dict[str, str]] = {}
    if args.workers == 1:
        for sequence in sequences:
            key, result = evaluate(sequence)
            cache[key] = result
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for key, result in executor.map(evaluate, sequences):
                cache[key] = result

    enriched: list[dict[str, str]] = []
    for row in rows:
        merged = dict(row)
        sequence = (row.get("sequence") or "").strip().upper().replace("T", "U")
        target = (row.get("target_dotbracket") or "").strip()
        if sequence and sequence in cache:
            merged.update(cache[sequence])
        else:
            merged.update(
                {
                    "viterbi_predicted_structure": "",
                    "viterbi_generated_structure": "",
                    "viterbi_energy": "",
                    "viterbi_runtime_sec": "",
                    "viterbi_error": "missing sequence",
                }
            )

        g = (row.get("target_g_dotbracket") or "").strip()
        gprime = (row.get("target_gprime_dotbracket") or "").strip()
        family = (row.get("target_gprime_family") or "").strip()
        if target and (not g or not gprime):
            g, gprime, family = split_target_dotbracket(target)
        merged["target_g_dotbracket"] = g
        merged["target_gprime_dotbracket"] = gprime
        merged["target_gprime_family"] = family or ""

        predicted = (merged.get("viterbi_predicted_structure") or "").strip()
        if predicted and target:
            try:
                distance = base_pair_distance(predicted, target)
                merged["viterbi_structure_distance"] = str(distance)
                normalized = normalized_distance(distance, sequence_length(sequence))
                merged["viterbi_structure_distance_normalized"] = "" if normalized is None else str(normalized)
            except Exception:
                merged["viterbi_structure_distance"] = ""
                merged["viterbi_structure_distance_normalized"] = ""
        else:
            merged["viterbi_structure_distance"] = ""
            merged["viterbi_structure_distance_normalized"] = ""
        enriched.append(merged)

    fieldnames = list(rows[0].keys())
    for result in enriched:
        for field in result:
            if field not in fieldnames:
                fieldnames.append(field)
    write_rows(args.output, enriched, fieldnames)
    errors = sum(bool((row.get("viterbi_error") or "").strip()) for row in enriched)
    print(f"Wrote {len(enriched)} rows to {args.output} ({errors} errors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
