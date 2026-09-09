#!/usr/bin/env python3
"""Assemble four methods x 20 candidates for common-G_big rescoring."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from evaluation_utils import ambiguous_base_count, dotbracket_to_pairs, normalize_sequence


METHODS = ("pkprobdesign", "desirna", "modena_ipknot", "antarna_pkiss")
COMPACT_CANDIDATE_FIELDS = (
    "pkb_id",
    "method",
    "rank",
    "sequence",
    "candidate_uid",
    "candidate_pool_occurrence",
    "modena_individual_id",
    "length",
    "target_dotbracket",
    "target_g_dotbracket",
    "target_gprime_dotbracket",
    "source_target_dotbracket",
    "scaffold_variant",
    "decomposition_source",
    "raw_source_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--desirna-root", type=Path, required=True)
    parser.add_argument("--external-normalized-root", type=Path, required=True)
    parser.add_argument("--modena-root", type=Path, required=True)
    parser.add_argument("--pkprobdesign-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--per-method", type=int, default=20)
    parser.add_argument("--modena-downsample-seed", type=int, default=20260629)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def downsample_modena(
    rows: list[dict[str, str]], *, pkb_id: str, count: int, seed: int
) -> list[dict[str, str]]:
    occurrences: defaultdict[str, int] = defaultdict(int)
    keyed: list[tuple[tuple[str, int], dict[str, str]]] = []
    for row in rows:
        sequence = normalize_sequence(row.get("sequence") or "")
        key = (sequence, occurrences[sequence])
        occurrences[sequence] += 1
        keyed.append((key, row))
    if len(keyed) < count:
        raise ValueError(f"{pkb_id}/modena_ipknot: only {len(keyed)} candidates")
    rng = random.Random(f"{seed}:modena_ipknot:{pkb_id}")
    selected_keys = set(rng.sample([key for key, _ in keyed], count))
    return [row for key, row in keyed if key in selected_keys]


def source_rows(
    *,
    target: dict[str, str],
    desirna_root: Path,
    external_normalized_root: Path,
    modena_root: Path,
    pkprobdesign_root: Path,
    per_method: int,
    modena_seed: int,
) -> dict[str, list[dict[str, str]]]:
    pkb_id = target["pkb_id"]
    desirna = read_csv(desirna_root / "normalized" / pkb_id / "candidates.csv")
    external = read_csv(external_normalized_root / pkb_id / "candidates.csv")
    antarna = [row for row in external if row.get("method") == "antarna_pkiss"]
    modena = read_csv(
        modena_root / "eval" / "normalized" / pkb_id / "big" / "candidates.csv"
    )
    modena = [row for row in modena if row.get("method") == "modena_ipknot"]
    pkprob = read_csv(
        pkprobdesign_root / "eval" / "normalized" / pkb_id / "big" / "candidates.csv"
    )
    pkprob = [row for row in pkprob if row.get("method") == "pkprobdesign"]
    pools = {
        "pkprobdesign": pkprob,
        "desirna": desirna,
        "modena_ipknot": downsample_modena(
            modena, pkb_id=pkb_id, count=per_method, seed=modena_seed
        ),
        "antarna_pkiss": antarna,
    }
    for method, rows in pools.items():
        if len(rows) != per_method:
            raise ValueError(
                f"{pkb_id}/{method}: expected {per_method} candidates, found {len(rows)}"
            )
    return pools


def align_pool(
    *, target: dict[str, str], method: str, rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    pkb_id = target["pkb_id"]
    length = int(target["length"])
    original_pairs = dotbracket_to_pairs(target["original_target"])
    recolored = target["recolored_target"]
    output: list[dict[str, str]] = []
    identity_occurrence: defaultdict[tuple[str, str], int] = defaultdict(int)
    for source_index, source in enumerate(rows, start=1):
        row = dict(source)
        sequence = normalize_sequence(row.get("sequence") or "")
        if len(sequence) != length or ambiguous_base_count(sequence):
            raise ValueError(f"{pkb_id}/{method}/{source_index}: invalid sequence")
        source_target = (row.get("target_dotbracket") or "").strip()
        if not source_target:
            raise ValueError(f"{pkb_id}/{method}/{source_index}: source target is missing")
        if dotbracket_to_pairs(source_target) != original_pairs:
            raise ValueError(f"{pkb_id}/{method}/{source_index}: source target pair map mismatch")
        rank = (row.get("rank") or str(source_index)).strip()
        occurrence_key = (rank, sequence)
        occurrence = identity_occurrence[occurrence_key]
        identity_occurrence[occurrence_key] += 1
        uid_payload = f"{pkb_id}|{method}|{rank}|{sequence}|{occurrence}"
        candidate_uid = hashlib.sha256(uid_payload.encode()).hexdigest()[:24]
        row.update(
            {
                "pkb_id": pkb_id,
                "method": method,
                "rank": rank,
                "sequence": sequence,
                "candidate_uid": candidate_uid,
                "candidate_pool_occurrence": str(occurrence),
                "source_target_dotbracket": source_target,
                "target_dotbracket": recolored,
                "original_target_dotbracket": target["original_target"],
                "target_g_dotbracket": target["g_big_dotbracket"],
                "target_gprime_dotbracket": target["g_small_dotbracket"],
                "target_gprime_family": "[]",
                "scaffold_variant": "big",
                "decomposition_source": "crossing_graph_maximum_g_big",
                "length": str(length),
            }
        )
        output.append(row)
    if len({row["candidate_uid"] for row in output}) != len(output):
        raise ValueError(f"{pkb_id}/{method}: candidate UID collision")
    return output


def main() -> int:
    args = parse_args()
    targets = read_csv(args.target_manifest)
    if len(targets) != 254:
        raise SystemExit(f"expected 254 targets, found {len(targets)}")
    all_rows: list[dict[str, str]] = []
    selected_modena: list[dict[str, str]] = []
    try:
        for target in targets:
            pools = source_rows(
                target=target,
                desirna_root=args.desirna_root,
                external_normalized_root=args.external_normalized_root,
                modena_root=args.modena_root,
                pkprobdesign_root=args.pkprobdesign_root,
                per_method=args.per_method,
                modena_seed=args.modena_downsample_seed,
            )
            target_rows: list[dict[str, str]] = []
            for method in METHODS:
                aligned = align_pool(target=target, method=method, rows=pools[method])
                target_rows.extend(aligned)
                if method == "modena_ipknot":
                    selected_modena.extend(
                        {
                            "pkb_id": target["pkb_id"],
                            "rank": row["rank"],
                            "sequence": row["sequence"],
                            "modena_individual_id": row.get("modena_individual_id", ""),
                            "candidate_uid": row["candidate_uid"],
                        }
                        for row in aligned
                    )
            if len(target_rows) != len(METHODS) * args.per_method:
                raise ValueError(f"{target['pkb_id']}: final candidate count mismatch")
            output = (
                args.results_root
                / "eval"
                / "normalized"
                / target["pkb_id"]
                / "big"
                / "candidates.csv"
            )
            write_csv(output, target_rows)
            all_rows.extend(target_rows)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error

    summary = args.results_root / "summary"
    compact_rows = [
        {key: row.get(key, "") for key in COMPACT_CANDIDATE_FIELDS}
        for row in all_rows
    ]
    write_csv(summary / "normalized_candidates.csv", compact_rows)
    write_csv(summary / "modena_downsampled_identities.csv", selected_modena)
    counts = {
        method: sum(row["method"] == method for row in all_rows) for method in METHODS
    }
    report = {
        "workflow": "common_gbig_four_method_candidate_assembly",
        "target_count": len(targets),
        "methods": list(METHODS),
        "candidates_per_target_method": args.per_method,
        "candidate_count": len(all_rows),
        "method_candidate_counts": counts,
        "method_target_group_count": len(targets) * len(METHODS),
        "method_target_groups_with_exactly_20": len(targets) * len(METHODS),
        "candidate_uid_unique_count": len({row["candidate_uid"] for row in all_rows}),
        "modena_source_candidates_per_target": 50,
        "modena_retained_candidates_per_target": args.per_method,
        "modena_downsample_seed": args.modena_downsample_seed,
        "target_manifest": str(args.target_manifest.resolve()),
        "desirna_root": str(args.desirna_root.resolve()),
        "external_normalized_root": str(args.external_normalized_root.resolve()),
        "modena_root": str(args.modena_root.resolve()),
        "pkprobdesign_root": str(args.pkprobdesign_root.resolve()),
        "all_checks_passed": (
            len(all_rows) == len(targets) * len(METHODS) * args.per_method
            and all(value == len(targets) * args.per_method for value in counts.values())
            and len({row["candidate_uid"] for row in all_rows}) == len(all_rows)
        ),
    }
    summary.mkdir(parents=True, exist_ok=True)
    (summary / "candidate_pool_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(all_rows)} common-G_big candidates under {args.results_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
