from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import evaluate_candidate_hotknots_viterbi as evaluator


class EvaluateCandidateHotKnotsViterbiTest(unittest.TestCase):
    def test_method_filter_is_case_insensitive_and_preserves_rows(self) -> None:
        rows = [
            {"method": "desirna", "sequence": "AAAA"},
            {"method": "pkprobdesign", "sequence": "CCCC"},
            {"method": "PKPROBDESIGN", "sequence": "GGGG"},
        ]
        selected = evaluator.filter_rows_by_methods(rows, ["PkProbDesign"])
        self.assertEqual([row["sequence"] for row in selected], ["CCCC", "GGGG"])

    def test_method_filter_rejects_missing_method(self) -> None:
        with self.assertRaisesRegex(ValueError, "no candidate rows matched"):
            evaluator.filter_rows_by_methods(
                [{"method": "desirna", "sequence": "AAAA"}], ["pkprobdesign"]
            )

    def test_parser_requires_one_selected_candidate(self) -> None:
        protocol = (
            "selected\tcandidate_index\tsource\thotknots_rank\talignment_score\t"
            "scaffold_structure\tunion_structure\tgenerated_structure\tenergy\terror\n"
            "1\t0\tempty\t\t\t.........\t(((...)))\t(((...)))\t-1.2\t\n"
            "0\t1\thotknots\t1\t760\t(((...)))\t(((...)))\t.........\t-1.2\t\n"
        )
        rows, selected = evaluator.parse_predictor_output(protocol)
        self.assertEqual(len(rows), 2)
        self.assertEqual(selected["source"], "empty")

    def test_evaluator_keeps_alignment_score_separate_from_energy(self) -> None:
        class Completed:
            returncode = 0
            stderr = ""
            stdout = (
                "selected\tcandidate_index\tsource\thotknots_rank\talignment_score\t"
                "scaffold_structure\tunion_structure\tgenerated_structure\tenergy\terror\n"
                "1\t1\thotknots\t1\t760\t(((...)))\t(((...)))\t.........\t-1.2\t\n"
            )

        with patch.object(evaluator.subprocess, "run", return_value=Completed()):
            result = evaluator.run_predictor(
                "GGGAAACCC",
                predictor_bin=Path("predictor"),
                wrapper_bin=Path("wrapper"),
                hotknots_root=Path("root"),
                param_file=Path("params"),
                hotspot_topk=20,
                hotspot_threshold=400,
            )
        self.assertEqual(result["hotknots_alignment_score"], "760")
        self.assertEqual(result["viterbi_energy"], "-1.2")
        self.assertEqual(result["viterbi_scaffold_energy"], "")


if __name__ == "__main__":
    unittest.main()
