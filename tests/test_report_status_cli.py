from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from hdfa_rl_suite import __version__
from hdfa_rl_suite.evaluation.report_status_cli import main, report_exit_code
from hdfa_rl_suite.simulator import SIMULATOR_VERSION


class ReportStatusCliTests(unittest.TestCase):
    def test_exit_code_contract(self) -> None:
        self.assertEqual(report_exit_code({"authoritative": True, "accepted": True}), 0)
        self.assertEqual(report_exit_code({"authoritative": True, "accepted": False}), 2)
        self.assertEqual(report_exit_code({"authoritative": False, "accepted": False}), 3)
        self.assertEqual(report_exit_code({"accepted": False}), 3)

    def test_main_reads_report_without_powershell_inline_quoting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(
                json.dumps({"authoritative": True, "accepted": False}),
                encoding="utf-8",
            )
            self.assertEqual(main([str(path)]), 2)

    def test_malformed_report_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text("not-json", encoding="utf-8")
            self.assertEqual(main([str(path)]), 3)

    def test_current_runtime_requirement_rejects_stale_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(json.dumps({
                "authoritative": True,
                "accepted": False,
                "provenance": {
                    "package_version": "stale",
                    "simulator_version": "stale",
                },
            }), encoding="utf-8")
            self.assertEqual(main([str(path), "--require-current-runtime"]), 3)
            path.write_text(json.dumps({
                "authoritative": True,
                "accepted": False,
                "provenance": {
                    "package_version": __version__,
                    "simulator_version": SIMULATOR_VERSION,
                },
            }), encoding="utf-8")
            self.assertEqual(main([str(path), "--require-current-runtime"]), 2)


if __name__ == "__main__":
    unittest.main()
