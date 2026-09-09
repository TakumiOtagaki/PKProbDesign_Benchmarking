#!/usr/bin/env python3
"""Select one representative per method using pseudo-joint probability only."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


METHODS = {"pkprobdesign", "desirna", "modena_ipknot", "antarna_pkiss"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-method", type=int, default=20)
    parser.add_argument("--analysis-label", default="common-G_big")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select_representatives(
    rows: list[dict[str, str]], *, per_method: int, analysis_label: str = "common-G_big"
) -> list[dict[str, str]]:
    methods = {row.get("method", "") for row in rows}
    if methods != METHODS:
        raise ValueError(f"unexpected method set: {sorted(methods)}")
    pkb_ids = {row.get("pkb_id", "") for row in rows}
    if len(pkb_ids) != 1:
        raise ValueError("input must contain exactly one target")
    selected: list[dict[str, str]] = []
    for method in sorted(METHODS):
        pool = [row for row in rows if row["method"] == method]
        if len(pool) != per_method:
            raise ValueError(f"{method}: expected {per_method} candidates, found {len(pool)}")
        scored: list[tuple[float, int, str, str, dict[str, str]]] = []
        for row in pool:
            error = (row.get("probability_error") or "").strip()
            if error:
                raise ValueError(f"{method}: probability error: {error}")
            try:
                score = float(row["combined_score"])
                logp_g = float(row["logP_G"])
                logp_branch = float(row["logP_G_prime_given_G_S"])
                rank = int(float(row["rank"]))
            except (KeyError, ValueError) as error:
                raise ValueError(f"{method}: invalid probability/rank field") from error
            values = (score, logp_g, logp_branch)
            if any(math.isnan(value) or value == math.inf for value in values):
                raise ValueError(f"{method}: invalid NaN or positive-infinite probability field")
            if any(value > 1e-8 for value in values):
                raise ValueError(f"{method}: log-probability field is greater than zero")
            scored.append((score, rank, row["sequence"], row["candidate_uid"], row))
        winner = dict(
            min(scored, key=lambda item: (-item[0], item[1], item[2], item[3]))[-1]
        )
        winner["representative_selection_rule"] = (
            f"maximum {analysis_label} pseudo-joint probability; "
            "ties: lower source rank, sequence, candidate_uid"
        )
        winner["selection_uses_target_distance"] = "false"
        winner["representative_probability_nonfinite"] = str(
            not math.isfinite(float(winner["combined_score"]))
        ).lower()
        winner["representative_nonfinite_reason"] = (
            "genuine_zero_probability_negative_infinity"
            if float(winner["combined_score"]) == -math.inf
            else ""
        )
        selected.append(winner)
    return sorted(selected, key=lambda row: row["method"])


def main() -> int:
    args = parse_args()
    try:
        selected = select_representatives(
            read_csv(args.input_csv),
            per_method=args.per_method,
            analysis_label=args.analysis_label,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    fields = list(selected[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)
    print(f"Selected {len(selected)} {args.analysis_label} representatives")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
