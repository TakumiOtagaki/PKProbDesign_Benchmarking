#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
import time
from pathlib import Path

from run_ipknot import build_command, parse_ipknot_output


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_IPKNOT_BIN = REPO_ROOT / "submodules" / "ipknot" / "ipknot.sif"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ipknot predictions for one PKB target candidate table."
    )
    parser.add_argument("--input-csv", type=Path, required=True, help="Normalized candidate CSV to enrich.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    parser.add_argument(
        "--binary",
        type=Path,
        default=DEFAULT_IPKNOT_BIN,
        help="Path to ipknot executable or SIF image.",
    )
    parser.add_argument(
        "--model",
        default="LinearPartition-C",
        help="Probability model for ipknot (-e).",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        help="Threshold(s) for -t. Repeatable.",
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


def run_ipknot(
    binary: Path,
    sequence: str,
    pkb_id: str,
    model: str,
    thresholds: list[str],
) -> dict[str, str]:
    start = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = Path(tmpdir) / "input.fa"
        fasta_path.write_text(f">{pkb_id}\n{sequence}\n", encoding="utf-8")
        cmd = build_command(str(binary), model, thresholds, fasta_path)
        result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "ipknot failed"
        return {
            "ipknot_predicted_structure": "",
            "ipknot_runtime_sec": f"{elapsed:.6f}",
            "ipknot_model": model,
            "ipknot_thresholds": ",".join(thresholds) if thresholds else "default",
            "ipknot_error": detail,
        }

    seq_out, struct_out = parse_ipknot_output(result.stdout)
    return {
        "ipknot_predicted_structure": struct_out,
        "ipknot_runtime_sec": f"{elapsed:.6f}",
        "ipknot_model": model,
        "ipknot_thresholds": ",".join(thresholds) if thresholds else "default",
        "ipknot_error": "",
        "ipknot_sequence": seq_out,
    }


def main() -> int:
    args = parse_args()
    rows = read_rows(args.input_csv)
    if not rows:
        raise SystemExit(f"No rows found in {args.input_csv}")

    cache: dict[str, dict[str, str]] = {}
    enriched_rows: list[dict[str, str]] = []
    for row in rows:
        sequence = (row.get("sequence") or "").strip().upper().replace("T", "U")
        pkb_id = (row.get("pkb_id") or "").strip()
        if sequence not in cache:
            cache[sequence] = run_ipknot(args.binary, sequence, pkb_id or "IPKNOT", args.model, args.threshold)
        merged = dict(row)
        merged.update(cache[sequence])
        enriched_rows.append(merged)

    fieldnames = list(rows[0].keys())
    for extra in [
        "ipknot_predicted_structure",
        "ipknot_runtime_sec",
        "ipknot_model",
        "ipknot_thresholds",
        "ipknot_error",
        "ipknot_sequence",
    ]:
        if extra not in fieldnames:
            fieldnames.append(extra)

    write_rows(args.output, enriched_rows, fieldnames)
    success_count = sum(
        bool((row.get("ipknot_predicted_structure") or "").strip())
        and not (row.get("ipknot_error") or "").strip()
        for row in enriched_rows
    )
    if success_count == 0:
        raise SystemExit(f"IPknot failed for all {len(enriched_rows)} candidate rows; see {args.output}")
    print(f"Wrote {len(enriched_rows)} rows to {args.output} ({success_count} IPknot predictions succeeded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
