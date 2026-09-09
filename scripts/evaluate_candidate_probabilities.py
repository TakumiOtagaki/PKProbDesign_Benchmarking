#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import os
import subprocess
from pathlib import Path

from evaluation_utils import combined_score_from_logs, split_target_dotbracket


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_LINEARPARTITION_BIN = REPO_ROOT / "bin" / "linearpartition_logprob"
DEFAULT_CPARTY_BIN = REPO_ROOT / "bin" / "scfg2_exact_adapter"
DEFAULT_CPARTY_PARAM_FILE = REPO_ROOT / "submodules" / "CParty" / "params" / "rna_DirksPierce09.par"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LinearPartition and CParty probabilities for one PKB target candidate table."
    )
    parser.add_argument("--input-csv", type=Path, required=True, help="Normalized candidate CSV to enrich.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument(
        "--linearpartition-bin",
        type=Path,
        default=DEFAULT_LINEARPARTITION_BIN,
        help="Path to the linearpartition_logprob helper.",
    )
    parser.add_argument(
        "--cparty-bin",
        type=Path,
        default=Path(os.environ.get("CPARTY_LOGPROB_BIN", str(DEFAULT_CPARTY_BIN))),
        help="Path to the SCFG2 exact adapter.",
    )
    parser.add_argument(
        "--cparty-param-file",
        type=Path,
        default=Path(os.environ.get("CPARTY_FIXED_ENERGY_PARAM_FILE", str(DEFAULT_CPARTY_PARAM_FILE))),
        help="Absolute CParty parameter file.",
    )
    parser.add_argument(
        "--beamsize",
        type=int,
        default=100,
        help="Beam size for the LinearPartition helper.",
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


def parse_float(text: str) -> float:
    value = (text or "").strip().split()
    if not value:
        raise ValueError("empty numeric output")
    return float(value[0])


def run_linearpartition(binary: Path, sequence: str, db: str, beamsize: int) -> float:
    cmd = [str(binary), sequence, db, str(beamsize)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "linearpartition_logprob failed"
        raise RuntimeError(detail)
    return parse_float(result.stdout)


def run_cparty(binary: Path, sequence: str, db_g: str, db_gprime: str, param_file: Path) -> float:
    cmd = [str(binary), "probability", sequence, db_g, db_gprime, str(param_file)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "CParty probability failed"
        raise RuntimeError(detail)
    return parse_float(result.stdout)


def evaluate_unique_sequence_target(
    sequence: str,
    target: str,
    g_dotbracket: str,
    gprime_dotbracket: str,
    *,
    linearpartition_bin: Path,
    cparty_bin: Path,
    cparty_param_file: Path,
    beamsize: int,
) -> tuple[tuple[str, str, str, str], dict[str, str]]:
    if not g_dotbracket or not gprime_dotbracket:
        g_dotbracket, gprime_dotbracket, gprime_family = split_target_dotbracket(target)
    else:
        gprime_family = "strict_density2"
    result: dict[str, str] = {
        "target_g_dotbracket": g_dotbracket,
        "target_gprime_dotbracket": gprime_dotbracket,
        "target_gprime_family": gprime_family or "",
        "logP_G": "",
        "logP_G_prime_given_G_S": "",
        "combined_score": "",
        "probability_error": "",
    }
    try:
        logp_g = run_linearpartition(linearpartition_bin, sequence, g_dotbracket, beamsize)
        result["logP_G"] = f"{logp_g:.12g}"
        gprime_has_pairs = any(ch in gprime_dotbracket for ch in "([{<")
        if gprime_has_pairs:
            logp_branch = run_cparty(cparty_bin, sequence, g_dotbracket, gprime_dotbracket, cparty_param_file)
            result["logP_G_prime_given_G_S"] = f"{logp_branch:.12g}"
            result["combined_score"] = f"{combined_score_from_logs(logp_g, logp_branch):.12g}"
        else:
            result["combined_score"] = f"{logp_g:.12g}"
    except Exception as exc:  # pragma: no cover - surfaced through row output
        result["probability_error"] = str(exc)
    return (sequence, target, g_dotbracket, gprime_dotbracket), result


def enrich_rows(
    rows: list[dict[str, str]],
    linearpartition_bin: Path,
    cparty_bin: Path,
    cparty_param_file: Path,
    beamsize: int,
    workers: int,
) -> list[dict[str, str]]:
    unique_pairs: list[tuple[str, str, str, str]] = []
    seen_pairs: set[tuple[str, str, str, str]] = set()
    for row in rows:
        sequence = (row.get("sequence") or "").strip().upper().replace("T", "U")
        target = (row.get("target_dotbracket") or "").strip()
        cache_key = (sequence, target, (row.get("target_g_dotbracket") or "").strip(), (row.get("target_gprime_dotbracket") or "").strip())
        if cache_key in seen_pairs:
            continue
        seen_pairs.add(cache_key)
        unique_pairs.append(cache_key)

    cache: dict[tuple[str, str, str, str], dict[str, str]] = {}
    if workers == 1:
        for sequence, target, g, gprime in unique_pairs:
            cache_key, result = evaluate_unique_sequence_target(
                sequence,
                target,
                g,
                gprime,
                linearpartition_bin=linearpartition_bin,
                cparty_bin=cparty_bin,
                cparty_param_file=cparty_param_file,
                beamsize=beamsize,
            )
            cache[cache_key] = result
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    evaluate_unique_sequence_target,
                    sequence,
                    target,
                    g,
                    gprime,
                    linearpartition_bin=linearpartition_bin,
                    cparty_bin=cparty_bin,
                    cparty_param_file=cparty_param_file,
                    beamsize=beamsize,
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
        cache_key = (sequence, target, (row.get("target_g_dotbracket") or "").strip(), (row.get("target_gprime_dotbracket") or "").strip())
        merged = dict(row)
        merged.update(cache[cache_key])
        enriched_rows.append(merged)

    return enriched_rows


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    rows = read_rows(args.input_csv)
    if not rows:
        raise SystemExit(f"No rows found in {args.input_csv}")

    enriched_rows = enrich_rows(
        rows,
        args.linearpartition_bin,
        args.cparty_bin,
        args.cparty_param_file,
        args.beamsize,
        args.workers,
    )

    fieldnames = list(rows[0].keys())
    for extra in [
        "target_g_dotbracket",
        "target_gprime_dotbracket",
        "target_gprime_family",
        "logP_G",
        "logP_G_prime_given_G_S",
        "combined_score",
        "probability_error",
    ]:
        if extra not in fieldnames:
            fieldnames.append(extra)

    write_rows(args.output, enriched_rows, fieldnames)
    print(f"Wrote {len(enriched_rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
