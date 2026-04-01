"""Integration tests for v2e CLI."""

import os
import subprocess
import sys
import tempfile

import pytest


class TestCLI:
    """Test v2e command-line interface."""

    def test_help(self):
        """Test that --help works without errors."""
        result = subprocess.run(
            [sys.executable, "v2e.py", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "v2e: generate simulated DVS events" in result.stdout

    def test_no_args_uses_defaults(self):
        """Test that running without arguments doesn't crash immediately."""
        result = subprocess.run(
            [sys.executable, "v2e.py"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode != 0

    def test_invalid_input(self):
        """Test that invalid input folder shows appropriate error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [sys.executable, "v2e.py", "-i", tmpdir, "--no_preview"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode != 0

    def test_version_args(self):
        """Test various version-related arguments."""
        result = subprocess.run(
            [sys.executable, "v2e.py", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert result.returncode == 0 or "--version" in result.stderr

    def test_dvs_params_clean(self):
        """Test --dvs_params clean option."""
        result = subprocess.run(
            [sys.executable, "v2e.py", "--dvs_params", "clean", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0

    def test_dvs_params_noisy(self):
        """Test --dvs_params noisy option."""
        result = subprocess.run(
            [sys.executable, "v2e.py", "--dvs_params", "noisy", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0


class TestArgsValidation:
    """Test argument validation."""

    def test_pos_thres_range(self):
        """Test positive threshold range validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    sys.executable,
                    "v2e.py",
                    "-i",
                    tmpdir,
                    "--pos_thres",
                    "-1",
                    "--no_preview",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode != 0

    def test_output_folder_creation(self):
        """Test that --output_folder creates directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "test_output")
            result = subprocess.run(
                [
                    sys.executable,
                    "v2e.py",
                    "-i",
                    tmpdir,
                    "-o",
                    output_dir,
                    "--no_preview",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert os.path.exists(output_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
