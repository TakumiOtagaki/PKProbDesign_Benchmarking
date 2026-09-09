#!/usr/bin/env python3
"""Reverse a validated common-G_big manifest for common-G_small analysis."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from evaluation_utils import dotbracket_to_pairs, pair_crosses, pairs_to_dotbracket


ALLOWED_DESIRNA_TARGET_CHARS = frozenset(".()[]")
ASSERTION_FIELDS = (
    "g_big_pkfree",
    "g_small_pkfree",
    "lanes_disjoint",
    "lanes_union_target",
    "original_common_pair_map_equal",
    "original_reversed_pair_map_equal",
    "common_reversed_pair_map_equal",
    "reversed_balanced",
    "desirna_syntax_accepted",
    "length_preserved",
    "position_coverage_preserved",
    "pair_counts_preserved",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-targets", type=int, default=254)
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


def merge_lanes(scaffold: str, extension: str) -> str:
    if len(scaffold) != len(extension):
        raise ValueError("scaffold and extension lengths differ")
    merged: list[str] = []
    for index, (scaffold_char, extension_char) in enumerate(
        zip(scaffold, extension)
    ):
        if scaffold_char != "." and extension_char != ".":
            raise ValueError(f"scaffold and extension overlap at nucleotide {index}")
        merged.append(scaffold_char if scaffold_char != "." else extension_char)
    return "".join(merged)


def reverse_target(source: dict[str, str]) -> dict[str, object]:
    required = {
        "pkb_id",
        "length",
        "natural_sequence",
        "original_target",
        "recolored_target",
        "g_big_dotbracket",
        "g_small_dotbracket",
    }
    missing = sorted(required - source.keys())
    if missing:
        raise ValueError(f"source manifest is missing columns: {', '.join(missing)}")

    pkb_id = source["pkb_id"]
    length = int(source["length"])
    original = source["original_target"]
    common_gbig = source["recolored_target"]
    natural_sequence = source["natural_sequence"]
    g_big_pairs = dotbracket_to_pairs(source["g_big_dotbracket"])
    g_small_pairs = dotbracket_to_pairs(source["g_small_dotbracket"])
    original_pairs = dotbracket_to_pairs(original)
    common_pairs = dotbracket_to_pairs(common_gbig)

    # Render each standalone component with parentheses, then render the
    # common-G_small combined target with G_small as () and G_big as [].
    g_big = pairs_to_dotbracket(g_big_pairs, length, "(", ")")
    g_small = pairs_to_dotbracket(g_small_pairs, length, "(", ")")
    g_small_scaffold = g_small
    g_big_extension = pairs_to_dotbracket(g_big_pairs, length, "[", "]")
    reversed_target = merge_lanes(g_small_scaffold, g_big_extension)
    reversed_pairs = dotbracket_to_pairs(reversed_target)

    assertions = {
        "g_big_pkfree": is_pkfree(g_big_pairs),
        "g_small_pkfree": is_pkfree(g_small_pairs),
        "lanes_disjoint": not bool(g_big_pairs & g_small_pairs),
        "lanes_union_target": (g_big_pairs | g_small_pairs) == original_pairs,
        "original_common_pair_map_equal": original_pairs == common_pairs,
        "original_reversed_pair_map_equal": original_pairs == reversed_pairs,
        "common_reversed_pair_map_equal": common_pairs == reversed_pairs,
        "reversed_balanced": reversed_pairs == dotbracket_to_pairs(reversed_target),
        "desirna_syntax_accepted": set(reversed_target)
        <= ALLOWED_DESIRNA_TARGET_CHARS,
        "length_preserved": (
            len(natural_sequence)
            == len(original)
            == len(common_gbig)
            == len(reversed_target)
            == len(g_big)
            == len(g_small)
            == length
        ),
        "position_coverage_preserved": set(range(length))
        == set(range(len(reversed_target))),
        "pair_counts_preserved": (
            len(original_pairs)
            == len(common_pairs)
            == len(reversed_pairs)
            == len(g_big_pairs) + len(g_small_pairs)
        ),
    }
    failed = [name for name, passed in assertions.items() if not passed]
    if failed:
        raise ValueError(f"{pkb_id}: failed assertions: {', '.join(failed)}")

    return {
        "pkb_id": pkb_id,
        "length": length,
        "natural_sequence": natural_sequence,
        "original_target": original,
        "common_gbig_target": common_gbig,
        "reversed_target": reversed_target,
        "g_big_dotbracket": g_big,
        "g_small_dotbracket": g_small,
        "g_small_scaffold_dotbracket": g_small_scaffold,
        "g_big_extension_dotbracket": g_big_extension,
        "target_pair_count": len(original_pairs),
        "g_big_pair_count": len(g_big_pairs),
        "g_small_pair_count": len(g_small_pairs),
        **assertions,
    }


def write_outputs(
    source_manifest: Path, output_dir: Path, records: list[dict[str, object]]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "reversed_targets.csv"
    joblist_path = output_dir / "reversed_joblist.txt"
    fields = (
        "pkb_id",
        "length",
        "natural_sequence",
        "original_target",
        "common_gbig_target",
        "reversed_target",
        "g_big_dotbracket",
        "g_small_dotbracket",
        "g_small_scaffold_dotbracket",
        "g_big_extension_dotbracket",
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
                f"{record['reversed_target']}|{record['length']}\n"
            )

    report = {
        "workflow": "common_gsmall_target_reversal",
        "source_of_truth": "validated common-G_big target manifest",
        "operation": "swap scaffold and extension bracket families without recomputing decomposition",
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest),
        "target_count": len(records),
        "unique_target_id_count": len({str(row["pkb_id"]) for row in records}),
        "assertion_fields": list(ASSERTION_FIELDS),
        "assertion_failure_count": 0,
        "all_assertions_passed": all(
            bool(row[field]) for row in records for field in ASSERTION_FIELDS
        ),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "reversed_joblist": str(joblist_path.resolve()),
        "reversed_joblist_sha256": sha256_file(joblist_path),
    }
    (output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    with args.source_manifest.open(encoding="utf-8", newline="") as handle:
        sources = list(csv.DictReader(handle))
    if len(sources) != args.expected_targets:
        raise SystemExit(
            f"expected {args.expected_targets} targets, found {len(sources)}"
        )
    ids = [source.get("pkb_id", "") for source in sources]
    if len(ids) != len(set(ids)) or not all(ids):
        raise SystemExit("source manifest has missing or duplicate target IDs")
    try:
        records = [reverse_target(source) for source in sources]
        write_outputs(args.source_manifest, args.output_dir, records)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"Validated and reversed {len(records)} targets under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
