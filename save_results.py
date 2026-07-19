"""
Utility: Save script results to output/results/ directory.

Provides:
  - TeeOutput: context manager that tees stdout to both console and file
  - save_csv: save DataFrame to CSV, creating dirs if needed
  - get_results_dir: resolve the output/results/ directory path
"""

import os
import sys
import io
import pandas as pd
from pathlib import Path


def get_results_dir() -> str:
    """Resolve the output/results/ directory path."""
    # From src/save_results.py -> project root -> output/results
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "output" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return str(results_dir)


def get_subdir(name: str) -> str:
    """Get (and create) a subdirectory under output/results/."""
    results_dir = get_results_dir()
    subdir = Path(results_dir) / name
    subdir.mkdir(parents=True, exist_ok=True)
    return str(subdir)


class TeeOutput:
    """
    Context manager that tees stdout to both console and a file.

    Usage:
        with TeeOutput("02_five_iterations", "five_iterations.txt"):
            print("This goes to both console and file")

    The file is written to output/results/<subdir>/<filename>.
    """

    def __init__(self, subdir: str, filename: str):
        self.subdir = subdir
        self.filename = filename
        self.file_path = None
        self._original_stdout = None
        self._tee = None

    def __enter__(self):
        subdir_path = get_subdir(self.subdir)
        self.file_path = os.path.join(subdir_path, self.filename)
        self._original_stdout = sys.stdout
        self._tee = _TeeWriter(self._original_stdout, self.file_path)
        sys.stdout = self._tee
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        if self._tee:
            self._tee.close()
        return False


class _TeeWriter:
    """File-like object that writes to both stdout and a file."""

    def __init__(self, stdout, file_path):
        self.stdout = stdout
        self.file = open(file_path, "w", encoding="utf-8")

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def save_csv(df: pd.DataFrame, subdir: str, filename: str) -> str:
    """
    Save a DataFrame to CSV under output/results/<subdir>/.

    Args:
        df: DataFrame to save
        subdir: subdirectory name (e.g. "02_five_iterations")
        filename: CSV filename (e.g. "iteration1_regression.csv")

    Returns:
        Full path to the saved file
    """
    subdir_path = get_subdir(subdir)
    file_path = os.path.join(subdir_path, filename)
    df.to_csv(file_path, index=False)
    return file_path
