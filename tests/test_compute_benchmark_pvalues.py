from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compute_benchmark_pvalues import wilcoxon_signed_rank


class WilcoxonSignedRankTest(unittest.TestCase):
    def test_strong_effect_has_nonzero_finite_tail_probability(self) -> None:
        x = pd.Series(range(1, 255), dtype=float)
        statistic, p_value, nonzero = wilcoxon_signed_rank(x, pd.Series(0.0, index=x.index))
        self.assertEqual(nonzero, 254)
        self.assertGreater(statistic, 0.0)
        self.assertTrue(math.isfinite(p_value))
        self.assertGreater(p_value, 0.0)
        self.assertLess(p_value, 1e-20)


if __name__ == "__main__":
    unittest.main()
