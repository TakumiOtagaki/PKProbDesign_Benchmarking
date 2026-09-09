from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from aggregate_common_gbig_four_method import (
    METHOD_ORDER,
    method_best_counts,
    paired_tests,
    validate_probability_pool,
    validate_representative_selection,
)


class AggregateCommonGbigFourMethodTest(unittest.TestCase):
    def records(self) -> pd.DataFrame:
        rows = []
        for target_index in range(3):
            for method_index, method in enumerate(METHOD_ORDER):
                rows.append(
                    {
                        "pkb_id": f"PKB{target_index}",
                        "method": method,
                        "combined_score": float(-method_index),
                        "logP_G": float(-method_index),
                        "logP_G_prime_given_G_S": float(-method_index),
                        "two_stage_structure_distance_normalized": float(method_index),
                        "viterbi_structure_distance_normalized": float(method_index),
                        "ipknot_structure_distance_normalized": float(method_index),
                    }
                )
        return pd.DataFrame(rows)

    def test_method_best_counts(self) -> None:
        counts = method_best_counts(self.records()).set_index("method")
        self.assertEqual(int(counts.loc["pkprobdesign", "unique_best_count"]), 3)
        self.assertEqual(int(counts["best_including_ties"].sum()), 3)

    def test_paired_tests_report_metric_and_global_bonferroni(self) -> None:
        tests = paired_tests(self.records())
        self.assertEqual(len(tests), 36)
        self.assertTrue(tests["bonferroni_metric_family_size"].eq(6).all())
        self.assertTrue(tests["bonferroni_global_family_size"].eq(36).all())
        self.assertTrue(
            (tests["p_value_bonferroni_global"] >= tests["p_value_bonferroni"]).all()
        )

    def test_probability_pool_and_selection_validation(self) -> None:
        rows = []
        for method in METHOD_ORDER:
            for rank in range(1, 21):
                rows.append(
                    {
                        "pkb_id": "PKB_TEST",
                        "method": method,
                        "rank": str(rank),
                        "sequence": f"ACGU{rank:02d}",
                        "candidate_uid": f"{method}-{rank}",
                        "logP_G": -float(rank),
                        "logP_G_prime_given_G_S": -1.0,
                        "combined_score": -float(rank + 1),
                        "probability_error": "",
                    }
                )
        probability = validate_probability_pool(pd.DataFrame(rows), expected_targets=1)
        selected = probability.loc[probability["rank"].eq("1")].copy()
        selected["selection_uses_target_distance"] = "false"
        validate_representative_selection(probability, selected)


if __name__ == "__main__":
    unittest.main()
