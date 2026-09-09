from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from select_common_gbig_representatives import METHODS, select_representatives


class SelectCommonGbigRepresentativesTest(unittest.TestCase):
    def test_selection_is_independent_of_target_distance(self) -> None:
        rows = []
        for method in METHODS:
            for rank in range(1, 21):
                rows.append(
                    {
                        "pkb_id": "PKB_TEST",
                        "method": method,
                        "rank": str(rank),
                        "sequence": ("A" * 11) + "C" + str(rank),
                        "candidate_uid": f"{method}-{rank}",
                        "logP_G": "-1",
                        "logP_G_prime_given_G_S": "-1",
                        "combined_score": str(-rank),
                        "probability_error": "",
                        "target_distance": str(21 - rank),
                    }
                )
        changed = copy.deepcopy(rows)
        for row in changed:
            row["target_distance"] = str(int(row["target_distance"]) * -100)
        first = select_representatives(rows, per_method=20)
        second = select_representatives(changed, per_method=20)
        self.assertEqual(
            [row["candidate_uid"] for row in first],
            [row["candidate_uid"] for row in second],
        )
        self.assertTrue(all(row["rank"] == "1" for row in first))

    def test_negative_infinity_is_retained_as_genuine_zero_probability(self) -> None:
        rows = []
        for method in METHODS:
            for rank in range(1, 21):
                rows.append(
                    {
                        "pkb_id": "PKB_TEST",
                        "method": method,
                        "rank": str(rank),
                        "sequence": "A" * (20 - rank) + "C" * rank,
                        "candidate_uid": f"{method}-{rank}",
                        "logP_G": "-inf",
                        "logP_G_prime_given_G_S": "-1",
                        "combined_score": "-inf",
                        "probability_error": "",
                    }
                )
        selected = select_representatives(
            rows, per_method=20, analysis_label="common-G_small"
        )
        self.assertTrue(
            all(row["representative_probability_nonfinite"] == "true" for row in selected)
        )
        self.assertTrue(
            all(
                row["representative_nonfinite_reason"]
                == "genuine_zero_probability_negative_infinity"
                for row in selected
            )
        )


if __name__ == "__main__":
    unittest.main()
