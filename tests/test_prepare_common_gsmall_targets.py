from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluation_utils import dotbracket_to_pairs
from prepare_common_gsmall_targets import reverse_target


def test_reverse_target_swaps_bracket_families_without_changing_pairs() -> None:
    source = {
        "pkb_id": "PKBTEST",
        "length": "12",
        "natural_sequence": "ACGUACGUACGU",
        "original_target": "((.[[.))..]]",
        "recolored_target": "((.[[.))..]]",
        "g_big_dotbracket": "((....))....",
        "g_small_dotbracket": "...[[.....]]",
    }

    result = reverse_target(source)

    assert result["reversed_target"] == "[[.((.]]..))"
    assert result["g_small_scaffold_dotbracket"] == "...((.....))"
    assert result["g_big_extension_dotbracket"] == "[[....]]...."
    assert dotbracket_to_pairs(str(result["original_target"])) == dotbracket_to_pairs(
        str(result["reversed_target"])
    )
    assert result["g_big_pair_count"] == 2
    assert result["g_small_pair_count"] == 2


def test_reverse_target_rejects_pair_map_drift() -> None:
    source = {
        "pkb_id": "PKBTEST",
        "length": "12",
        "natural_sequence": "ACGUACGUACGU",
        "original_target": "((.[[.))..]]",
        "recolored_target": "((..[[))..]]",
        "g_big_dotbracket": "((....))....",
        "g_small_dotbracket": "...[[.....]]",
    }

    try:
        reverse_target(source)
    except ValueError as error:
        assert "failed assertions" in str(error)
    else:
        raise AssertionError("pair-map drift was not rejected")


def load_tests(loader, tests, pattern):
    return unittest.TestSuite([
        unittest.FunctionTestCase(test_reverse_target_swaps_bracket_families_without_changing_pairs),
        unittest.FunctionTestCase(test_reverse_target_rejects_pair_map_drift),
    ])
