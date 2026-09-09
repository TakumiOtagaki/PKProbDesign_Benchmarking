from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from normalize_common_gbig_desirna_target import normalize_rows


class NormalizeCommonGbigDesirnaTargetTest(unittest.TestCase):
    def test_normalizes_exact_requested_prefix(self) -> None:
        target = {
            "length": "12",
            "original_target": "(([[..))..]]",
            "recolored_target": "[[((..]]..))",
            "g_big_dotbracket": "..((......))",
            "g_small_dotbracket": "[[....]]....",
        }
        source = [
            {
                "sequence": "ACGUACGUACGU",
                "mfe_ss": "[[((..]]..))",
                "Epf": "-1.0",
                "scoring_function": "0.5",
                "mcc": "0.25",
            }
            for _ in range(21)
        ]
        rows = normalize_rows(
            pkb_id="PKB_TEST",
            target=target,
            source_rows=source,
            source_path=Path("source.csv"),
            runtime_sec=60.0,
            expected_candidates=20,
        )
        self.assertEqual(len(rows), 20)
        self.assertEqual(rows[0]["target_dotbracket"], target["recolored_target"])
        self.assertEqual(rows[0]["target_g_dotbracket"], target["g_big_dotbracket"])

    def test_normalizes_common_gsmall_orientation(self) -> None:
        target = {
            "length": "12",
            "original_target": "(([[..))..]]",
            "reversed_target": "[[((..]]..))",
            "g_small_scaffold_dotbracket": "..((......))",
            "g_big_extension_dotbracket": "[[....]]....",
        }
        source = [
            {
                "sequence": "ACGUACGUACGU",
                "mfe_ss": "[[((..]]..))",
                "Epf": "-1.0",
                "scoring_function": "0.5",
                "mcc": "0.25",
            }
            for _ in range(20)
        ]
        rows = normalize_rows(
            pkb_id="PKB_TEST",
            target=target,
            source_rows=source,
            source_path=Path("source.csv"),
            runtime_sec=60.0,
            expected_candidates=20,
            design_target_column="reversed_target",
            scaffold_column="g_small_scaffold_dotbracket",
            extension_column="g_big_extension_dotbracket",
            scaffold_variant="small",
            decomposition_source="crossing_graph_complement_g_small_scaffold",
        )
        self.assertEqual(rows[0]["target_dotbracket"], target["reversed_target"])
        self.assertEqual(
            rows[0]["target_g_dotbracket"], target["g_small_scaffold_dotbracket"]
        )
        self.assertEqual(rows[0]["scaffold_variant"], "small")


if __name__ == "__main__":
    unittest.main()
