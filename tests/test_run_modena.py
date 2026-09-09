import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "run_modena.py"
SPEC = importlib.util.spec_from_file_location("run_modena", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
run_modena = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_modena)


def make_args(**overrides):
    values = {
        "binary": "/opt/modena",
        "backend": "rnafold",
        "iterations": None,
        "outint": None,
        "seed": None,
        "threads": None,
        "extra_arg": [],
        "input_file": "/tmp/target.inp",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class BuildCommandTests(unittest.TestCase):
    def test_ipknot_backend_is_passed_as_native_modena_flag(self):
        args = make_args(backend="ipknot", iterations=3, seed=7, threads=2)
        self.assertEqual(
            run_modena.build_command(args),
            [
                "/opt/modena",
                "-it",
                "3",
                "-r",
                "7",
                "-mp",
                "2",
                "-ipknot",
                "-f",
                "/tmp/target.inp",
            ],
        )

    def test_rnafold_backend_adds_no_backend_flag(self):
        args = make_args(backend="rnafold")
        self.assertEqual(
            run_modena.build_command(args),
            ["/opt/modena", "-f", "/tmp/target.inp"],
        )


class BackendTraceTests(unittest.TestCase):
    def test_ipknot_backend_requires_trace_path(self):
        with self.assertRaisesRegex(RuntimeError, "requires --backend-trace-file"):
            run_modena.verify_backend_trace("ipknot", run_modena.summarize_trace(None))

    def test_successful_trace_is_accepted_without_banner_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "calls.tsv"
            trace.write_text(
                "start\t101\t\t2026-07-22T00:00:00Z\n"
                "complete\t101\t0\t2026-07-22T00:00:01Z\n",
                encoding="utf-8",
            )
            summary = run_modena.summarize_trace(str(trace))
            run_modena.verify_backend_trace("ipknot", summary)
            self.assertEqual(summary["successful_completions"], 1)

    def test_missing_ipknot_invocation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "calls.tsv"
            with self.assertRaisesRegex(RuntimeError, "without invoking"):
                run_modena.verify_backend_trace(
                    "ipknot", run_modena.summarize_trace(str(trace))
                )

    def test_ipknot_wrapper_records_a_completed_process(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "calls.tsv"
            env = os.environ.copy()
            env["IPKNOT_BIN"] = shutil.which("true") or "/usr/bin/true"
            env["IPKNOT_TRACE_FILE"] = str(trace)
            subprocess.run(
                [str(REPO_ROOT / "scripts" / "wrappers" / "ipknot"), "input.fa"],
                check=True,
                env=env,
            )
            summary = run_modena.summarize_trace(str(trace))
            self.assertEqual(summary["starts"], 1)
            self.assertEqual(summary["successful_completions"], 1)
            run_modena.verify_backend_trace("ipknot", summary)

    def test_ipknot_sif_wrapper_binds_current_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            fake_apptainer = root / "fake-apptainer"
            argument_log = root / "arguments.txt"
            fake_apptainer.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$@\" > \"$ARGUMENT_LOG\"\n",
                encoding="utf-8",
            )
            fake_apptainer.chmod(0o755)
            env = os.environ.copy()
            env["APPTAINER_BIN"] = os.fspath(fake_apptainer)
            env["ARGUMENT_LOG"] = os.fspath(argument_log)
            env["IPKNOT_BIN"] = os.fspath(root / "ipknot.sif")
            subprocess.run(
                [str(REPO_ROOT / "scripts" / "wrappers" / "ipknot"), "modena.0.sq"],
                check=True,
                cwd=root,
                env=env,
            )
            arguments = argument_log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                arguments,
                [
                    "exec",
                    "--bind",
                    f"{root}:{root}",
                    "--pwd",
                    os.fspath(root),
                    os.fspath(root / "ipknot.sif"),
                    "ipknot",
                    "modena.0.sq",
                ],
            )

    def test_runner_accepts_successful_trace_when_banner_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            fake_modena = root / "fake_modena"
            fake_modena.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "ipknot ignored-input.fa >/dev/null\n"
                "printf 'Individual= 0 Rk= 1 Sc= 1.0 1.0\\n\\nAAAA\\n....\\n'\n",
                encoding="utf-8",
            )
            fake_modena.chmod(0o755)
            input_file = root / "target.inp"
            input_file.write_text("....\n\n;\n", encoding="utf-8")
            trace = root / "calls.tsv"
            metadata = root / "metadata.json"
            result = root / "result.csv"
            env = os.environ.copy()
            env["IPKNOT_BIN"] = shutil.which("true") or "/usr/bin/true"
            env["PATH"] = f"{REPO_ROOT / 'scripts' / 'wrappers'}:{env['PATH']}"

            subprocess.run(
                [
                    os.fspath(MODULE_PATH),
                    "--binary",
                    os.fspath(fake_modena),
                    "--input-file",
                    os.fspath(input_file),
                    "--backend",
                    "ipknot",
                    "--backend-trace-file",
                    os.fspath(trace),
                    "--run-metadata",
                    os.fspath(metadata),
                    "--output-csv",
                    os.fspath(result),
                ],
                check=True,
                env=env,
            )
            evidence = json.loads(metadata.read_text(encoding="utf-8"))
            self.assertEqual(evidence["backend_banner"], "")
            self.assertEqual(evidence["trace"]["successful_completions"], 1)

    def test_failed_ipknot_invocation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "calls.tsv"
            trace.write_text(
                "start\t101\t\t2026-07-22T00:00:00Z\n"
                "complete\t101\t2\t2026-07-22T00:00:01Z\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "failed calls"):
                run_modena.verify_backend_trace(
                    "ipknot", run_modena.summarize_trace(str(trace))
                )


if __name__ == "__main__":
    unittest.main()
