#!/usr/bin/env python3
"""Tests for deploy-tool.sh: syntax, usage, and the files it deploys."""

import json
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
DEPLOY_SCRIPT = REPO_ROOT / "deploy-tool.sh"


class TestDeployScript(unittest.TestCase):
    def test_exists_and_is_executable(self):
        self.assertTrue(DEPLOY_SCRIPT.exists())
        self.assertTrue(DEPLOY_SCRIPT.stat().st_mode & 0o111)

    def test_bash_syntax(self):
        result = subprocess.run(
            ["bash", "-n", str(DEPLOY_SCRIPT)], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_usage_on_an_unknown_option(self):
        result = subprocess.run(
            ["bash", str(DEPLOY_SCRIPT), "--invalid-option"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Usage:", result.stdout + result.stderr)

    def test_pyproject_declares_no_console_scripts(self):
        """The tool is invoked as a file, so pyproject must not add scripts."""
        text = (REPO_ROOT / "pyproject.toml").read_text()
        self.assertNotIn("[project.scripts]", text)

    def test_route_example_is_a_single_route(self):
        example = json.loads(
            (REPO_ROOT / "tool" / "ocr_routing.json.example").read_text()
        )
        self.assertIn("model", example)
        self.assertIn("base_url", example)


if __name__ == "__main__":
    unittest.main()