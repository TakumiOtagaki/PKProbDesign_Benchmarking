#!/usr/bin/env python3
import argparse
import csv
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run antaRNA for a single structure and emit a normalized TSV."
    )
    parser.add_argument("--structure", required=True, help="Target dot-bracket structure.")
    parser.add_argument("--pkb-id", required=True, help="Target identifier.")
    parser.add_argument("--gc", type=float, default=0.5, help="Target GC content.")
    parser.add_argument("--count", type=int, default=1, help="Number of sequences to output (-n).")
    parser.add_argument("--time", type=int, default=600, help="Runtime limit in seconds (-t).")
    parser.add_argument(
        "--pseudoknots",
        action="store_true",
        help="Enable pseudoknot mode (-p).",
    )
    parser.add_argument(
        "--pkprogram",
        default="pKiss",
        help="Pseudoknot program (pKiss|HotKnots|IPKnot).",
    )
    parser.add_argument(
        "--python",
        default="python",
        help="Python interpreter to run antaRNA (likely python2).",
    )
    parser.add_argument(
        "--binary",
        default="submodules/antarna/antarna/antarna.py",
        help="Path to antaRNA script.",
    )
    parser.add_argument(
        "--output-tsv",
        required=True,
        help="Output TSV path.",
    )
    parser.add_argument(
        "--raw-output",
        default=None,
        help="Optional path to save raw antaRNA stdout.",
    )
    return parser.parse_args()


def parse_antarna_output(text: str) -> list[tuple[int, str, str, str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    results = []
    i = 0
    rank = 1
    while i < len(lines):
        header = lines[i]
        if not header.startswith(">"):
            i += 1
            continue
        if i + 1 >= len(lines):
            break
        seq_line = lines[i + 1]
        struct_line = lines[i + 2] if i + 2 < len(lines) else ""

        sequence = seq_line.replace("Rseq:", "") if seq_line.startswith("Rseq:") else seq_line
        structure = struct_line.replace("Rstr:", "") if struct_line.startswith("Rstr:") else ""

        results.append((rank, sequence, structure, header))
        rank += 1
        i += 3
    if not results:
        raise RuntimeError("No antaRNA results found in output")
    return results


def main() -> int:
    args = parse_args()

    cmd = [
        args.python,
        args.binary,
        "-tGC",
        str(args.gc),
        "-t",
        str(args.time),
        "-n",
        str(args.count),
        "--name",
        args.pkb_id,
        "-ov",
        "MFE",
        "-Cstr",
        args.structure,
    ]
    if args.pseudoknots:
        cmd += ["-p", "-pkP", args.pkprogram]

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "antaRNA failed"
        raise RuntimeError(detail)

    if args.raw_output:
        raw_path = Path(args.raw_output)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(result.stdout, encoding="utf-8")

    records = parse_antarna_output(result.stdout)

    output_path = Path(args.output_tsv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "pkb_id",
                "rank",
                "sequence",
                "predicted_structure",
                "header",
                "runtime_sec",
            ]
        )
        for rank, sequence, structure, header in records:
            writer.writerow(
                [
                    args.pkb_id,
                    rank,
                    sequence,
                    structure,
                    header,
                    f"{elapsed:.6f}",
                ]
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
