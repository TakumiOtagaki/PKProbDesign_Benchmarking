from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from aggregate_common_gsmall_four_method import (
    METHOD_ORDER,
    assignment_sensitivity_tests,
    classify_probability_pool,
    method_best_counts,
    paired_tests,
)


class AggregateCommonGsmallFourMethodTest(unittest.TestCase):
    def records(self) -> pd.DataFrame:
        rows = []
        for target_index in range(3):
            for method_index, method in enumerate(METHOD_ORDER):
                score = -float(method_index)
                if target_index == 2:
                    score = -np.inf
                rows.append(
                    {
                        "pkb_id": f"PKB{target_index}",
                        "method": method,
                        "combined_score": score,
                        "logP_G": score,
                        "logP_G_prime_given_G_S": score,
                        "two_stage_structure_distance_normalized": float(method_index),
                        "viterbi_structure_distance_normalized": float(method_index),
                        "ipknot_structure_distance_normalized": float(method_index),
                    }
                )
        return pd.DataFrame(rows)

    def test_negative_infinity_is_counted_as_tied_best_when_all_methods_zero(self) -> None:
        counts = method_best_counts(self.records()).set_index("method")
        self.assertEqual(int(counts.loc["pkprobdesign", "unique_best_count"]), 2)
        self.assertTrue(counts["tied_best_count"].eq(1).all())

    def test_probability_tests_report_finite_pair_count_and_bonferroni(self) -> None:
        tests = paired_tests(self.records())
        probability = tests.loc[tests["metric"].eq("combined_score")]
        self.assertTrue(probability["paired_targets_total"].eq(3).all())
        self.assertTrue(probability["paired_targets_finite"].eq(2).all())
        self.assertTrue(probability["paired_targets_excluded_nonfinite"].eq(1).all())
        self.assertTrue(
            probability["bonferroni_six_comparison_family_size"].eq(6).all()
        )

    def test_negative_infinity_is_classified_when_empty_error_reads_as_nan(self) -> None:
        rows = []
        for method in METHOD_ORDER:
            for rank in range(1, 21):
                rows.append(
                    {
                        "pkb_id": "PKB_TEST",
                        "method": method,
                        "rank": rank,
                        "candidate_uid": f"{method}-{rank}",
                        "logP_G": -np.inf if rank == 1 else -1.0,
                        "logP_G_prime_given_G_S": -1.0,
                        "combined_score": -np.inf if rank == 1 else -2.0,
                        "probability_error": np.nan,
                    }
                )
        _, nonfinite, errors = classify_probability_pool(
            pd.DataFrame(rows), expected_targets=1
        )
        self.assertEqual(len(nonfinite), len(METHOD_ORDER) * 2)
        self.assertTrue(errors.empty)
        self.assertTrue(nonfinite["classification"].eq("genuine_zero_probability").all())

    def test_assignment_sensitivity_tests_do_not_claim_same_probability(self) -> None:
        rows = []
        for method in ("pkprobdesign", "desirna"):
            for target in range(4):
                for assignment, offset in (("big", 0.0), ("small", 1.0)):
                    rows.append(
                        {
                            "pkb_id": f"PKB{target}",
                            "method": method,
                            "evaluation_scaffold_assignment": assignment,
                            "logP_scaffold": -3.0 + offset,
                            "logP_extension_given_scaffold": -2.0 - offset,
                            "log_pseudo_joint": -5.0,
                        }
                    )
        tests = assignment_sensitivity_tests(pd.DataFrame(rows))
        self.assertEqual(len(tests), 6)
        self.assertTrue(tests["paired_targets_finite"].eq(4).all())
        self.assertTrue(tests["scores_are_assignment_specific"].all())
        self.assertFalse(tests["same_full_target_probability_claimed"].any())
        self.assertFalse(tests["oracle_selection_performed"].any())


if __name__ == "__main__":
    unittest.main()
