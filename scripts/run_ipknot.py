#!/usr/bin/env python3
import argparse
import csv
import subprocess
import tempfile
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ipknot on a single sequence and emit a normalized TSV row."
    )
    parser.add_argument("--sequence", required=True, help="RNA sequence (A/C/G/U).")
    parser.add_argument("--pkb-id", required=True, help="Target identifier.")
    parser.add_argument("--binary", default="ipknot", help="Path to ipknot binary.")
    parser.add_argument(
        "--model",
        default="LinearPartition-C",
        help="Probability model for ipknot (-e).",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        help="Threshold(s) for -t. Repeatable. If omitted, ipknot defaults are used.",
    )
    parser.add_argument(
        "--output-tsv",
        required=True,
        help="Output TSV path.",
    )
    parser.add_argument(
        "--raw-output",
        default=None,
        help="Optional path to save raw ipknot stdout.",
    )
    return parser.parse_args()


def build_command(binary: str, model: str, thresholds: list[str], fasta_path: Path) -> list[str]:
    if binary.endswith(".sif"):
        cmd = ["apptainer", "exec", binary, "ipknot"]
    else:
        cmd = [binary]
    if model:
        cmd += ["-e", model]
    for threshold in thresholds:
        cmd += ["-t", threshold]
    cmd.append(str(fasta_path))
    return cmd


def parse_ipknot_output(text: str) -> tuple[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        raise RuntimeError("ipknot output did not contain 3 lines")
    if not lines[0].startswith(">"):
        raise RuntimeError("ipknot output missing FASTA header")
    sequence = lines[1]
    structure = lines[2]
    return sequence, structure


def main() -> int:
    args = parse_args()
    start = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = Path(tmpdir) / "input.fa"
        fasta_path.write_text(f">{args.pkb_id}\n{args.sequence}\n", encoding="utf-8")

        cmd = build_command(args.binary, args.model, args.threshold, fasta_path)

        result = subprocess.run(cmd, capture_output=True, text=True)

    elapsed = time.time() - start
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "ipknot failed"
        raise RuntimeError(detail)

    if args.raw_output:
        raw_path = Path(args.raw_output)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(result.stdout, encoding="utf-8")

    seq_out, struct_out = parse_ipknot_output(result.stdout)

    output_path = Path(args.output_tsv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "pkb_id",
                "sequence",
                "predicted_structure",
                "model",
                "thresholds",
                "runtime_sec",
            ]
        )
        writer.writerow(
            [
                args.pkb_id,
                seq_out,
                struct_out,
                args.model,
                ",".join(args.threshold) if args.threshold else "default",
                f"{elapsed:.6f}",
            ]
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
