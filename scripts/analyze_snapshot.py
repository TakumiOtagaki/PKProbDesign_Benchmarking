#!/usr/bin/env python3
"""Validate and reaggregate the published data without folding software or HPC."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import aggregate_common_gbig_four_method as big
import aggregate_common_gsmall_four_method as small
from evaluation_utils import base_pair_distance, dotbracket_to_pairs, maximal_strict_density2_decomposition


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_structures(records: pd.DataFrame) -> None:
    for row in records.to_dict("records"):
        target = row["target_dotbracket"]
        length = int(row["length"])
        require(len(target) == len(row["sequence"]) == length, "Length mismatch")
        require(set(row["sequence"]) <= set("ACGU"), "Invalid designed sequence")
        for prefix in ("two_stage", "viterbi", "ipknot"):
            error = row[f"{prefix}_error"]
            require(pd.isna(error) or error == "", f"{prefix}: prediction error")
            prediction = row[f"{prefix}_predicted_structure"]
            require(len(prediction) == length, f"{prefix}: prediction length mismatch")
            distance = base_pair_distance(target, prediction)
            require(distance == row[f"{prefix}_structure_distance"], f"{prefix}: distance mismatch")
            require(np.isclose(distance / length, row[f"{prefix}_structure_distance_normalized"],
                               rtol=1e-10, atol=1e-12), f"{prefix}: normalized distance mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--output", type=Path, default=Path("output/snapshot"))
    args = parser.parse_args()
    data = args.data_dir.resolve()
    out = args.output.resolve()
    require(not out.is_relative_to(data) and not data.is_relative_to(out),
            "Output must be separate from the published data directory")
    require(not out.exists() or not any(out.iterdir()), "Refusing to overwrite nonempty output")
    manifest = json.loads((data / "manifest.json").read_text())
    for name, metadata in manifest["files"].items():
        path = (data / name).resolve()
        require(path.is_relative_to(data), "Invalid manifest path")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == metadata["export_sha256"],
                f"Checksum mismatch: {name}")
    targets = pd.read_csv(data / "targets.csv")
    require(len(targets) == 254 and targets["pkb_id"].nunique() == 254, "Expected 254 targets")
    require(targets["original_target"].nunique() == 254, "Target structures are not deduplicated")
    require(targets["target_index"].tolist() == list(range(1, 255)), "Invalid target index order")
    target_ids = set(targets["pkb_id"])
    for row in targets.to_dict("records"):
        original = dotbracket_to_pairs(row["original_target"])
        require(len(row["original_target"]) == row["length"] <= 100, "Invalid target length")
        require(all(j - i - 1 >= 3 for i, j in original), "Minimum-loop filter violated")
        require(original == dotbracket_to_pairs(row["recolored_target"]), "Recoloring changed pairs")
        decomposition = maximal_strict_density2_decomposition(original)
        require(decomposition == (dotbracket_to_pairs(row["g_big_dotbracket"]),
                                  dotbracket_to_pairs(row["g_small_dotbracket"])),
                "Target decomposition mismatch")
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"checksums_verified": len(manifest["files"]), "target_count": 254}
    for assignment, module in (("big", big), ("small", small)):
        folder = data / f"common_g{assignment}"
        pool = pd.read_csv(folder / "candidate_pool.csv", float_precision="round_trip")
        records = pd.read_csv(folder / "representatives.csv", float_precision="round_trip")
        require(set(pool["pkb_id"]) == set(records["pkb_id"]) == target_ids, "Target coverage mismatch")
        if assignment == "big":
            pool = module.validate_probability_pool(pool, 254)
        else:
            pool, _, errors = module.classify_probability_pool(pool, 254)
            require(errors.empty, "Probability evaluation errors")
        records = module.validate_selected(records, 254)
        module.validate_representative_selection(pool, records)
        # Compare the full exported probability row, not just the selected UID.
        columns = list(pool.columns)
        expected = pool.loc[pool["candidate_uid"].isin(records["candidate_uid"]), columns]
        pd.testing.assert_frame_equal(
            records[columns].sort_values("candidate_uid").reset_index(drop=True),
            expected.sort_values("candidate_uid").reset_index(drop=True),
            check_dtype=False, check_exact=True,
        )
        target_map = targets.set_index("pkb_id")
        for row in pool.to_dict("records"):
            target = target_map.loc[row["pkb_id"]]
            scaffold = "g_big_dotbracket" if assignment == "big" else "g_small_dotbracket"
            extension = "g_small_dotbracket" if assignment == "big" else "g_big_dotbracket"
            require(dotbracket_to_pairs(row["target_dotbracket"]) == dotbracket_to_pairs(target["original_target"]), "Pool target mismatch")
            require(dotbracket_to_pairs(row["target_g_dotbracket"]) == dotbracket_to_pairs(target[scaffold]), "Pool scaffold mismatch")
            require(dotbracket_to_pairs(row["target_gprime_dotbracket"]) == dotbracket_to_pairs(target[extension]), "Pool extension mismatch")
            require(len(row["sequence"]) == row["length"] == target["length"] and set(row["sequence"]) <= set("ACGU"), "Invalid pool sequence")
        require(np.allclose(pool["logP_G"] + pool["logP_G_prime_given_G_S"], pool["combined_score"],
                            rtol=1e-9, atol=1e-9), "Probability terms do not sum to combined score")
        check_structures(records)
        destination = out / f"common_g{assignment}"
        destination.mkdir()
        for name in ("method_summary", "method_best_counts", "paired_tests"):
            computed = getattr(module, name)(records)
            reference = pd.read_csv(folder / "reference" / f"{name}.csv", float_precision="round_trip")
            pd.testing.assert_frame_equal(computed.reset_index(drop=True), reference,
                                          check_dtype=False, rtol=1e-9, atol=1e-300)
            computed.to_csv(destination / f"{name}.csv", index=False)
        counts = module.method_best_counts(records).set_index("method")["unique_best_count"].to_dict()
        report[f"common_g{assignment}"] = {
            "candidate_count": len(pool), "representative_count": len(records),
            "selection_reproduced": True, "structure_distances_reproduced": True,
            "reference_tables_reproduced": True, "unique_best_counts": counts,
        }
    (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
