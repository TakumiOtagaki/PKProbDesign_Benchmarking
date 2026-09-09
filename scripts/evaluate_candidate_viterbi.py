#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import os
import subprocess
import tempfile
import time
from pathlib import Path

from evaluation_utils import base_pair_distance, split_target_dotbracket
from evaluation_utils import normalized_distance, sequence_length


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_VITERBI_BIN = REPO_ROOT / "bin" / "scfg2_exact_adapter"
DEFAULT_VITERBI_PARAM_FILE = REPO_ROOT / "submodules" / "CParty" / "params" / "rna_DirksPierce09.par"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate CParty Viterbi predictions for one PKB target candidate table."
    )
    parser.add_argument("--input-csv", type=Path, required=True, help="Normalized candidate CSV to enrich.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument(
        "--binary",
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
        "--workers",
        type=int,
        default=1,
        help="Number of worker threads for unique (sequence, target) evaluations.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_viterbi_output(text: str) -> tuple[str, str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("viterbi output was empty")
    first = lines[0].split()
    if not first:
        raise RuntimeError("viterbi output missing structure token")
    structure = first[0]
    energy = first[1] if len(first) > 1 else ""
    generated = first[2] if len(first) > 2 else ""
    return structure, energy, generated


def run_viterbi(binary: Path, sequence: str, g_dotbracket: str, param_file: Path) -> dict[str, str]:
    start = time.time()
    cmd = [str(binary), "viterbi", sequence, g_dotbracket]
    if param_file:
        cmd.append(str(param_file))
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "viterbi failed"
        return {
            "viterbi_predicted_structure": "",
            "viterbi_generated_structure": "",
            "viterbi_energy": "",
            "viterbi_runtime_sec": f"{elapsed:.6f}",
            "viterbi_error": detail,
        }

    try:
        structure, energy, generated = parse_viterbi_output(result.stdout)
    except Exception as exc:  # pragma: no cover - surfaced through row output
        return {
            "viterbi_predicted_structure": "",
            "viterbi_generated_structure": "",
            "viterbi_energy": "",
            "viterbi_runtime_sec": f"{elapsed:.6f}",
            "viterbi_error": str(exc),
        }

    return {
        "viterbi_predicted_structure": structure,
        "viterbi_generated_structure": generated,
        "viterbi_energy": energy,
        "viterbi_runtime_sec": f"{elapsed:.6f}",
        "viterbi_error": "",
    }


def evaluate_unique_sequence_target(
    sequence: str,
    target: str,
    g_dotbracket: str,
    gprime_dotbracket: str,
    *,
    binary: Path,
    param_file: Path,
) -> tuple[tuple[str, str, str, str], dict[str, str]]:
    if not g_dotbracket or not gprime_dotbracket:
        g_dotbracket, gprime_dotbracket, gprime_family = split_target_dotbracket(target)
    else:
        gprime_family = "strict_density2"
    result = run_viterbi(binary, sequence, g_dotbracket, param_file)
    result.update(
        {
            "target_g_dotbracket": g_dotbracket,
            "target_gprime_dotbracket": gprime_dotbracket,
            "target_gprime_family": gprime_family or "",
        }
    )
    return (sequence, target, g_dotbracket, gprime_dotbracket), result


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    rows = read_rows(args.input_csv)
    if not rows:
        raise SystemExit(f"No rows found in {args.input_csv}")

    unique_pairs: list[tuple[str, str, str, str]] = []
    seen_pairs: set[tuple[str, str, str, str]] = set()
    for row in rows:
        sequence = (row.get("sequence") or "").strip().upper().replace("T", "U")
        target = (row.get("target_dotbracket") or "").strip()
        if not sequence or not target:
            continue
        cache_key = (sequence, target, (row.get("target_g_dotbracket") or "").strip(), (row.get("target_gprime_dotbracket") or "").strip())
        if cache_key in seen_pairs:
            continue
        seen_pairs.add(cache_key)
        unique_pairs.append(cache_key)

    cache: dict[tuple[str, str, str, str], dict[str, str]] = {}
    if args.workers == 1:
        for sequence, target, g, gprime in unique_pairs:
            cache_key, result = evaluate_unique_sequence_target(
                sequence,
                target,
                g,
                gprime,
                binary=args.binary,
                param_file=args.param_file,
            )
            cache[cache_key] = result
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    evaluate_unique_sequence_target,
                    sequence,
                    target,
                    g,
                    gprime,
                    binary=args.binary,
                    param_file=args.param_file,
                )
                for sequence, target, g, gprime in unique_pairs
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
            merged.update(
                {
                    "target_g_dotbracket": "",
                    "target_gprime_dotbracket": "",
                    "target_gprime_family": "",
                    "viterbi_predicted_structure": "",
                    "viterbi_generated_structure": "",
                    "viterbi_energy": "",
                    "viterbi_runtime_sec": "",
                    "viterbi_error": "missing sequence or target_dotbracket",
                    "viterbi_structure_distance": "",
                }
            )
            enriched_rows.append(merged)
            continue

        cache_key = (sequence, target, (row.get("target_g_dotbracket") or "").strip(), (row.get("target_gprime_dotbracket") or "").strip())
        merged = dict(row)
        merged.update(cache[cache_key])
        if merged.get("viterbi_predicted_structure") and target:
            try:
                distance = base_pair_distance(merged["viterbi_predicted_structure"], target)
                merged["viterbi_structure_distance"] = str(distance)
                normalized = normalized_distance(distance, sequence_length(sequence))
                merged["viterbi_structure_distance_normalized"] = "" if normalized is None else str(normalized)
            except Exception:
                merged["viterbi_structure_distance"] = ""
                merged["viterbi_structure_distance_normalized"] = ""
        else:
            merged["viterbi_structure_distance"] = ""
            merged["viterbi_structure_distance_normalized"] = ""
        enriched_rows.append(merged)

    fieldnames = list(rows[0].keys())
    for extra in [
        "target_g_dotbracket",
        "target_gprime_dotbracket",
        "target_gprime_family",
        "viterbi_predicted_structure",
        "viterbi_generated_structure",
        "viterbi_energy",
        "viterbi_runtime_sec",
        "viterbi_error",
        "viterbi_structure_distance",
        "viterbi_structure_distance_normalized",
    ]:
        if extra not in fieldnames:
            fieldnames.append(extra)

    write_rows(args.output, enriched_rows, fieldnames)
    print(f"Wrote {len(enriched_rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
