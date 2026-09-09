#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import subprocess
import time
from pathlib import Path

from evaluation_utils import base_pair_distance, normalized_distance, sequence_length
from evaluate_candidate_viterbi import (
    REPO_ROOT,
    DEFAULT_VITERBI_BIN,
    DEFAULT_VITERBI_PARAM_FILE,
    read_rows,
    run_viterbi,
    write_rows,
)

DEFAULT_LINEARFOLD_BIN = REPO_ROOT / "bin" / "linearfold_predict"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate two-stage predictions by inferring a pk-free scaffold with LinearFold before CParty Viterbi extension."
    )
    parser.add_argument("--input-csv", type=Path, required=True, help="Normalized candidate CSV to enrich.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument(
        "--viterbi-bin",
        type=Path,
        default=Path(os.environ.get("CPARTY_VITERBI_BIN", str(DEFAULT_VITERBI_BIN))),
        help="Path to the SCFG2 exact adapter.",
    )
    parser.add_argument(
        "--param-file",
        type=Path,
        default=Path(os.environ.get("CPARTY_VITERBI_PARAM_FILE", str(DEFAULT_VITERBI_PARAM_FILE))),
        help="Absolute CParty parameter file.",
    )
    parser.add_argument(
        "--linearfold-bin",
        type=Path,
        default=Path(os.environ.get("LINEARFOLD_PREDICT_BIN", str(DEFAULT_LINEARFOLD_BIN))),
        help="Path to the LinearFold scaffold predictor used for two-stage seed inference.",
    )
    parser.add_argument(
        "--beamsize",
        type=int,
        default=100,
        help="Beam size for the LinearFold scaffold predictor.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker threads for unique sequence evaluations.",
    )
    return parser.parse_args()


def run_linearfold_seed(binary: Path, sequence: str, beamsize: int) -> dict[str, str]:
    start = time.time()
    cmd = [str(binary), sequence, str(beamsize)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "LinearFold seed prediction failed"
        return {
            "two_stage_seed_structure_union": "",
            "two_stage_seed_energy": "",
            "two_stage_seed_g_dotbracket": "",
            "two_stage_seed_gprime_dotbracket": "",
            "two_stage_seed_runtime_sec": f"{elapsed:.6f}",
            "two_stage_seed_error": detail,
        }

    try:
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("LinearFold output is empty")
        structure_union = lines[0]
        energy = lines[1] if len(lines) > 1 else ""
        if len(structure_union) != len(sequence):
            raise RuntimeError("LinearFold output length mismatch")
    except Exception as exc:  # pragma: no cover
        return {
            "two_stage_seed_structure_union": "",
            "two_stage_seed_energy": "",
            "two_stage_seed_g_dotbracket": "",
            "two_stage_seed_gprime_dotbracket": "",
            "two_stage_seed_runtime_sec": f"{elapsed:.6f}",
            "two_stage_seed_error": str(exc),
        }

    return {
        "two_stage_seed_structure_union": structure_union,
        "two_stage_seed_energy": energy,
        "two_stage_seed_g_dotbracket": structure_union,
        "two_stage_seed_gprime_dotbracket": "." * len(sequence),
        "two_stage_seed_runtime_sec": f"{elapsed:.6f}",
        "two_stage_seed_error": "",
    }


def blank_result(error: str) -> dict[str, str]:
    return {
        "two_stage_seed_structure_union": "",
        "two_stage_seed_energy": "",
        "two_stage_seed_g_dotbracket": "",
        "two_stage_seed_gprime_dotbracket": "",
        "two_stage_seed_runtime_sec": "",
        "two_stage_seed_error": error,
        "two_stage_predicted_structure": "",
        "two_stage_generated_structure": "",
        "two_stage_energy": "",
        "two_stage_runtime_sec": "",
        "two_stage_error": error,
        "two_stage_structure_distance": "",
        "two_stage_structure_distance_normalized": "",
    }


def evaluate_unique_sequence(
    sequence: str,
    *,
    linearfold_bin: Path,
    viterbi_bin: Path,
    param_file: Path,
    beamsize: int,
) -> tuple[str, dict[str, str]]:
    seed = run_linearfold_seed(linearfold_bin, sequence, beamsize)
    if seed["two_stage_seed_error"] or not seed["two_stage_seed_g_dotbracket"]:
        return sequence, {
            **seed,
            **blank_result(seed["two_stage_seed_error"] or "two-stage seed scaffold missing"),
        }
    second = run_viterbi(viterbi_bin, sequence, seed["two_stage_seed_g_dotbracket"], param_file)
    return sequence, {
        **seed,
        "two_stage_predicted_structure": second.get("viterbi_predicted_structure", ""),
        "two_stage_generated_structure": second.get("viterbi_generated_structure", ""),
        "two_stage_energy": second.get("viterbi_energy", ""),
        "two_stage_runtime_sec": second.get("viterbi_runtime_sec", ""),
        "two_stage_error": second.get("viterbi_error", ""),
    }


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    rows = read_rows(args.input_csv)
    if not rows:
        raise SystemExit(f"No rows found in {args.input_csv}")

    unique_sequences: list[str] = []
    seen_sequences: set[str] = set()
    for row in rows:
        sequence = (row.get("sequence") or "").strip().upper().replace("T", "U")
        if not sequence or sequence in seen_sequences:
            continue
        seen_sequences.add(sequence)
        unique_sequences.append(sequence)

    cache: dict[str, dict[str, str]] = {}
    if args.workers == 1:
        for sequence in unique_sequences:
            cache_key, result = evaluate_unique_sequence(
                sequence,
                linearfold_bin=args.linearfold_bin,
                viterbi_bin=args.viterbi_bin,
                param_file=args.param_file,
                beamsize=args.beamsize,
            )
            cache[cache_key] = result
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    evaluate_unique_sequence,
                    sequence,
                    linearfold_bin=args.linearfold_bin,
                    viterbi_bin=args.viterbi_bin,
                    param_file=args.param_file,
                    beamsize=args.beamsize,
                )
                for sequence in unique_sequences
            ]
            for future in futures:
                cache_key, result = future.result()
                cache[cache_key] = result

    enriched_rows: list[dict[str, str]] = []
    for row in rows:
        sequence = (row.get("sequence") or "").strip().upper().replace("T", "U")
        target = (row.get("target_dotbracket") or "").strip()
        if not sequence or not target:
            merged = dict(row)
            merged.update(blank_result("missing sequence or target_dotbracket"))
            enriched_rows.append(merged)
            continue

        merged = dict(row)
        merged.update(cache[sequence])
        predicted = (merged.get("two_stage_predicted_structure") or "").strip()
        if predicted and target and not merged.get("two_stage_error"):
            try:
                distance = base_pair_distance(predicted, target)
                merged["two_stage_structure_distance"] = str(distance)
                normalized = normalized_distance(distance, sequence_length(sequence))
                merged["two_stage_structure_distance_normalized"] = "" if normalized is None else str(normalized)
            except Exception:
                merged["two_stage_structure_distance"] = ""
                merged["two_stage_structure_distance_normalized"] = ""
        else:
            merged["two_stage_structure_distance"] = ""
            merged["two_stage_structure_distance_normalized"] = ""
        enriched_rows.append(merged)

    fieldnames = list(rows[0].keys())
    for extra in [
        "two_stage_seed_structure_union",
        "two_stage_seed_energy",
        "two_stage_seed_g_dotbracket",
        "two_stage_seed_gprime_dotbracket",
        "two_stage_seed_runtime_sec",
        "two_stage_seed_error",
        "two_stage_predicted_structure",
        "two_stage_generated_structure",
        "two_stage_energy",
        "two_stage_runtime_sec",
        "two_stage_error",
        "two_stage_structure_distance",
        "two_stage_structure_distance_normalized",
    ]:
        if extra not in fieldnames:
            fieldnames.append(extra)

    write_rows(args.output, enriched_rows, fieldnames)
    print(f"Wrote {len(enriched_rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
