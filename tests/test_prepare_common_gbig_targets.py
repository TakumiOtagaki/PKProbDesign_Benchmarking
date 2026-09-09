from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluation_utils import dotbracket_to_pairs
from prepare_common_gbig_targets import merge_lanes, recolor_target


class PrepareCommonGbigTargetsTest(unittest.TestCase):
    def test_recolor_preserves_pairs_and_maximizes_big_lane(self) -> None:
        pairs = {(0, 3), (1, 4), (5, 10), (6, 12), (11, 13), (14, 16)}
        chars = ["."] * 17
        families = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]
        for (left, right), (open_char, close_char) in zip(
            sorted(pairs), families * 2
        ):
            chars[left] = open_char
            chars[right] = close_char
        source = {
            "pkb_id": "PKB_TEST",
            "natural_sequence": "A" * 17,
            "original_target": "".join(chars),
            "length": 17,
        }

        record = recolor_target(source)

        self.assertEqual(record["g_big_pair_count"], 4)
        self.assertEqual(record["g_small_pair_count"], 2)
        self.assertEqual(
            dotbracket_to_pairs(str(record["original_target"])),
            dotbracket_to_pairs(str(record["recolored_target"])),
        )
        self.assertEqual(set(str(record["recolored_target"])) - set(".()[]"), set())

    def test_merge_lanes_rejects_endpoint_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            merge_lanes("(..)", "[..]")


if __name__ == "__main__":
    unittest.main()
