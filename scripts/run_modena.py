#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MODENA on an input .inp file and convert its DesignedSequences output to CSV."
    )
    parser.add_argument(
        "--input-file",
        "-f",
        required=True,
        help="Path to a MODENA input file.",
    )
    parser.add_argument(
        "--binary",
        default="submodules/modena0067b_macos10.9/modena",
        help="Path to the MODENA executable.",
    )
    parser.add_argument(
        "--backend",
        choices=("rnafold", "ipknot", "hotknots"),
        default="rnafold",
        help="Structure-prediction backend passed to MODENA [default: rnafold].",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override MODENA -it.",
    )
    parser.add_argument(
        "--outint",
        type=int,
        default=None,
        help="Override MODENA -outint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override MODENA -r.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Override MODENA -mp.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional argument to pass to MODENA. Repeatable.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV output path. Defaults to stdout when omitted.",
    )
    parser.add_argument(
        "--raw-output",
        default=None,
        help="Optional path to save raw MODENA stdout.",
    )
    parser.add_argument(
        "--backend-trace-file",
        default=None,
        help=(
            "Optional wrapper trace file. For the ipknot backend, the runner exports "
            "IPKNOT_TRACE_FILE and requires at least one successful traced call."
        ),
    )
    parser.add_argument(
        "--run-metadata",
        default=None,
        help="Optional JSON path recording the command and backend verification evidence.",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    cmd = [args.binary]
    if args.iterations is not None:
        cmd += ["-it", str(args.iterations)]
    if args.outint is not None:
        cmd += ["-outint", str(args.outint)]
    if args.seed is not None:
        cmd += ["-r", str(args.seed)]
    if args.threads is not None:
        cmd += ["-mp", str(args.threads)]
    if args.backend == "ipknot":
        cmd.append("-ipknot")
    elif args.backend == "hotknots":
        cmd.append("-hotknots")
    for extra in args.extra_arg:
        cmd.append(extra)
    cmd += ["-f", args.input_file]
    return cmd


def run_modena(args: argparse.Namespace) -> tuple[str, float, list[str]]:
    cmd = build_command(args)
    env = os.environ.copy()
    if args.backend_trace_file:
        trace_path = Path(args.backend_trace_file)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.unlink(missing_ok=True)
        env["IPKNOT_TRACE_FILE"] = str(trace_path.resolve())

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = time.time() - start
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or "modena failed"
        raise RuntimeError(detail)
    return result.stdout, elapsed, cmd


def summarize_trace(path: str | None) -> dict[str, object]:
    summary: dict[str, object] = {
        "path": path or "",
        "starts": 0,
        "completions": 0,
        "successful_completions": 0,
        "failed_completions": 0,
    }
    if not path:
        return summary

    trace_path = Path(path)
    if not trace_path.is_file():
        return summary

    for line in trace_path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if not fields:
            continue
        if fields[0] == "start":
            summary["starts"] = int(summary["starts"]) + 1
        elif fields[0] == "complete":
            summary["completions"] = int(summary["completions"]) + 1
            status = fields[2] if len(fields) > 2 else ""
            key = "successful_completions" if status == "0" else "failed_completions"
            summary[key] = int(summary[key]) + 1
    return summary


def verify_backend_trace(backend: str, trace: dict[str, object]) -> None:
    if backend != "ipknot":
        return
    if not trace.get("path"):
        raise RuntimeError("the ipknot backend requires --backend-trace-file")
    starts = int(trace["starts"])
    completions = int(trace["completions"])
    successes = int(trace["successful_completions"])
    failures = int(trace["failed_completions"])
    if starts == 0:
        raise RuntimeError("MODENA completed without invoking the traced ipknot wrapper")
    if starts != completions:
        raise RuntimeError(
            f"ipknot trace is incomplete: starts={starts}, completions={completions}"
        )
    if failures or successes != starts:
        raise RuntimeError(
            f"ipknot trace contains failed calls: starts={starts}, successes={successes}, failures={failures}"
        )


def parse_float(value: str):
    try:
        return float(value)
    except ValueError:
        return value


def parse_score_line(record: dict[str, object], line: str) -> None:
    match = re.match(r"Individual=\s*(\d+)\s+Rk=\s*(\d+)\s+Sc=\s*(.*)", line)
    if not match:
        raise ValueError(f"Unrecognized MODENA score line: {line}")
    record["individual_id"] = int(match.group(1))
    record["rank"] = int(match.group(2))
    score_values = [parse_float(token) for token in match.group(3).split()]
    for idx, value in enumerate(score_values, start=1):
        record[f"score_{idx}"] = value


def parse_metric_line(record: dict[str, object], line: str) -> bool:
    tokens = line.split()
    if len(tokens) < 2:
        return False
    if len(tokens[0]) != 1 or not tokens[0].isalpha():
        return False

    label = tokens[0]
    record[f"{label}_tool"] = tokens[1]
    for token in tokens[2:]:
        if ":" not in token:
            continue
        key, value = token.split(":", 1)
        record[f"{label}_{key.lower()}"] = parse_float(value)
    return True


def parse_structure_line(record: dict[str, object], line: str) -> bool:
    tokens = line.split()
    if not tokens:
        return False
    structure = tokens[0]
    if not all(ch in ".()[]{}<>" for ch in structure):
        return False
    record["predicted_structure"] = structure
    for token in tokens[1:]:
        if ":" not in token:
            continue
        key, value = token.split(":", 1)
        record[key.lower()] = value
    return True


def normalize_key(name: str) -> str:
    key = name.strip().lower()
    key = key.replace(">3", "_gt_3")
    key = key.replace(">", "_gt_")
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return key.strip("_")


def parse_assignment_line(record: dict[str, object], line: str) -> bool:
    matches = re.findall(r"([^=\s][^=]*?)=\s*([^\s]+)", line)
    if not matches:
        return False
    for key, value in matches:
        record[normalize_key(key)] = parse_float(value)
    return True


def parse_modena_output(output: str, elapsed: float, input_file: str) -> list[dict[str, object]]:
    lines = output.splitlines()
    records: list[dict[str, object]] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line.startswith("Individual="):
            index += 1
            continue

        record: dict[str, object] = {
            "input_file": input_file,
            "time": elapsed,
        }
        parse_score_line(record, line)

        index += 1
        while index < len(lines) and not lines[index].strip():
            index += 1
        if index >= len(lines):
            break
        record["sequence"] = lines[index].strip()
        index += 1

        while index < len(lines):
            current = lines[index].strip()
            if not current:
                index += 1
                break
            if current.startswith("Individual="):
                break
            if parse_metric_line(record, current):
                index += 1
                continue
            if parse_structure_line(record, current):
                index += 1
                continue
            parse_assignment_line(record, current)
            index += 1

        records.append(record)

    if not records:
        raise RuntimeError("No DesignedSequences blocks found in MODENA output.")
    return records


def write_csv(records: list[dict[str, object]], output_csv: str | None) -> None:
    fieldnames = sorted({key for record in records for key in record.keys()})
    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        handle = output_path.open("w", encoding="utf-8", newline="")
        close_handle = True
    else:
        handle = sys.stdout
        close_handle = False

    try:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    finally:
        if close_handle:
            handle.close()


def main() -> int:
    args = parse_args()
    stdout, elapsed, cmd = run_modena(args)
    trace = summarize_trace(args.backend_trace_file)
    verify_backend_trace(args.backend, trace)

    banner_backend = ""
    for backend, marker in (
        ("ipknot", "IPknot is used."),
        ("hotknots", "HotKnots is used."),
        ("rnafold", "RNAfold is used."),
    ):
        if marker in stdout:
            banner_backend = backend
            break

    print(f"MODENA command: {shlex.join(cmd)}", file=sys.stderr)
    print(
        f"MODENA backend: requested={args.backend}, banner={banner_backend or 'not-detected'}, "
        f"traced_successes={trace['successful_completions']}",
        file=sys.stderr,
    )

    if args.raw_output:
        raw_path = Path(args.raw_output)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(stdout, encoding="utf-8")

    records = parse_modena_output(stdout, elapsed, args.input_file)
    for record in records:
        record["backend_requested"] = args.backend
        record["backend_banner"] = banner_backend
        record["backend_traced_successes"] = trace["successful_completions"]
    write_csv(records, args.output_csv)

    if args.run_metadata:
        metadata_path = Path(args.run_metadata)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(
                {
                    "backend_requested": args.backend,
                    "backend_banner": banner_backend,
                    "command": cmd,
                    "elapsed_seconds": elapsed,
                    "trace": trace,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
