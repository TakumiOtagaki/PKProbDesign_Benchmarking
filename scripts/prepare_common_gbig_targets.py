#!/usr/bin/env python3
"""Build and validate the common-G_big recolored DesiRNA target manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from evaluation_utils import (
    dotbracket_to_pairs,
    maximal_strict_density2_decomposition,
    normalize_sequence,
    pair_crosses,
    pairs_to_dotbracket,
    validate_base_pair_matching,
)


ALLOWED_DESIRNA_TARGET_CHARS = frozenset(".()[]")
ASSERTION_FIELDS = (
    "g_big_pkfree",
    "g_small_pkfree",
    "lanes_disjoint",
    "lanes_union_target",
    "pair_map_preserved",
    "recolored_balanced",
    "desirna_syntax_accepted",
    "length_preserved",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joblist", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_pkfree(pairs: frozenset[tuple[int, int]]) -> bool:
    ordered = sorted(pairs)
    return not any(
        pair_crosses(left, right)
        for index, left in enumerate(ordered)
        for right in ordered[index + 1 :]
    )


def merge_lanes(g_big: str, g_small: str) -> str:
    if len(g_big) != len(g_small):
        raise ValueError("G_big and G_small lengths differ")
    merged: list[str] = []
    for index, (big_char, small_char) in enumerate(zip(g_big, g_small)):
        if big_char != "." and small_char != ".":
            raise ValueError(f"G_big and G_small overlap at nucleotide {index}")
        merged.append(big_char if big_char != "." else small_char)
    return "".join(merged)


def parse_joblist(path: Path) -> list[dict[str, object]]:
    targets: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split("|", 3)
            if len(fields) != 4:
                raise ValueError(
                    f"{path}:{line_number}: expected four pipe-delimited fields"
                )
            pkb_id, sequence, target, length_text = fields
            if pkb_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate target ID {pkb_id}")
            seen_ids.add(pkb_id)
            length = int(length_text)
            normalized_sequence = normalize_sequence(sequence)
            if len(normalized_sequence) != length or len(target) != length:
                raise ValueError(
                    f"{pkb_id}: sequence/target length does not match declared length"
                )
            targets.append(
                {
                    "pkb_id": pkb_id,
                    "natural_sequence": normalized_sequence,
                    "original_target": target,
                    "length": length,
                }
            )
    return targets


def recolor_target(source: dict[str, object]) -> dict[str, object]:
    pkb_id = str(source["pkb_id"])
    original = str(source["original_target"])
    length = int(source["length"])
    original_pairs = dotbracket_to_pairs(original)
    validate_base_pair_matching(original_pairs)
    decomposition = maximal_strict_density2_decomposition(original_pairs)
    if decomposition is None:
        raise ValueError(f"{pkb_id}: target is not strict density-2")
    g_big_pairs, g_small_pairs = decomposition
    g_big = pairs_to_dotbracket(g_big_pairs, length, "(", ")")
    g_small = pairs_to_dotbracket(g_small_pairs, length, "[", "]")
    recolored = merge_lanes(g_big, g_small)

    recolored_pairs = dotbracket_to_pairs(recolored)
    assertions = {
        "g_big_pkfree": is_pkfree(g_big_pairs),
        "g_small_pkfree": is_pkfree(g_small_pairs),
        "lanes_disjoint": not bool(g_big_pairs & g_small_pairs),
        "lanes_union_target": (g_big_pairs | g_small_pairs) == original_pairs,
        "pair_map_preserved": recolored_pairs == original_pairs,
        "recolored_balanced": recolored_pairs == dotbracket_to_pairs(recolored),
        "desirna_syntax_accepted": set(recolored) <= ALLOWED_DESIRNA_TARGET_CHARS,
        "length_preserved": len(g_big) == len(g_small) == len(recolored) == length,
    }
    failed = [name for name, passed in assertions.items() if not passed]
    if failed:
        raise ValueError(f"{pkb_id}: failed assertions: {', '.join(failed)}")

    return {
        **source,
        "recolored_target": recolored,
        "g_big_dotbracket": g_big,
        "g_small_dotbracket": g_small,
        "target_pair_count": len(original_pairs),
        "g_big_pair_count": len(g_big_pairs),
        "g_small_pair_count": len(g_small_pairs),
        **assertions,
    }


def write_outputs(
    *,
    source_joblist: Path,
    output_dir: Path,
    records: list[dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "recolored_targets.csv"
    joblist_path = output_dir / "recolored_joblist.txt"
    fields = (
        "pkb_id",
        "length",
        "natural_sequence",
        "original_target",
        "recolored_target",
        "g_big_dotbracket",
        "g_small_dotbracket",
        "target_pair_count",
        "g_big_pair_count",
        "g_small_pair_count",
        *ASSERTION_FIELDS,
    )
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    with joblist_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                f"{record['pkb_id']}|{record['natural_sequence']}|"
                f"{record['recolored_target']}|{record['length']}\n"
            )

    report = {
        "workflow": "common_gbig_target_recoloring",
        "decomposition": "crossing_graph_component_bipartition_maximum_g_big",
        "tie_rule": "lexicographically_smaller_colour_class_to_g_big",
        "source_joblist": str(source_joblist.resolve()),
        "source_joblist_sha256": sha256_file(source_joblist),
        "target_count": len(records),
        "recolored_target_count": len(records),
        "unique_target_id_count": len({str(row["pkb_id"]) for row in records}),
        "assertion_fields": list(ASSERTION_FIELDS),
        "assertion_failure_count": 0,
        "all_assertions_passed": all(
            bool(row[field]) for row in records for field in ASSERTION_FIELDS
        ),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "recolored_joblist": str(joblist_path.resolve()),
        "recolored_joblist_sha256": sha256_file(joblist_path),
    }
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    sources = parse_joblist(args.joblist)
    if len(sources) != 254:
        raise SystemExit(f"expected 254 targets, found {len(sources)}")
    try:
        records = [recolor_target(source) for source in sources]
        write_outputs(
            source_joblist=args.joblist,
            output_dir=args.output_dir,
            records=records,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(f"Validated and recolored {len(records)} targets under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
