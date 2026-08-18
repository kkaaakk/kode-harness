"""Tests for agents/code_search.py — grep_search and glob_search."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.code_search import (
    glob_search,
    grep_search,
    SEARCH_EXCLUDE_DIRS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    """Create a minimal workspace with a few searchable files."""
    # Python files
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text(
        "import os\n"
        "def main():\n"
        "    print('hello world')\n"
        "    return 0\n",
        encoding="utf-8",
    )
    (src / "utils.py").write_text(
        "def helper(x):\n"
        "    return x * 2\n"
        "def MAIN():\n"
        "    pass\n",
        encoding="utf-8",
    )

    # Text file
    (tmp_path / "README.md").write_text(
        "# Project\nThis is a demo project.\n",
        encoding="utf-8",
    )

    # Excluded dir content
    venv = tmp_path / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "noise.py").write_text("hello world\n", encoding="utf-8")

    pycache = tmp_path / "src" / "__pycache__"
    pycache.mkdir()
    (pycache / "cached.pyc").write_bytes(b"hello world")

    return tmp_path


# ---------------------------------------------------------------------------
# grep_search tests
# ---------------------------------------------------------------------------


class TestGrepSearch:
    def test_finds_pattern_in_files(self, workspace: Path) -> None:
        result = grep_search("hello", workdir=workspace)
        text = str(result)
        assert "src/main.py:3:" in text
        assert "hello world" in text
        assert "matches" in text

    def test_regex_pattern(self, workspace: Path) -> None:
        result = grep_search(r"def \w+\(", workdir=workspace)
        text = str(result)
        assert "main()" in text
        assert "helper(" in text

    def test_empty_pattern_returns_error(self, workspace: Path) -> None:
        result = grep_search("", workdir=workspace)
        text = str(result)
        assert text.startswith("Error:")

    def test_invalid_regex_returns_error(self, workspace: Path) -> None:
        result = grep_search("[invalid", workdir=workspace)
        text = str(result)
        assert text.startswith("Error:")

    def test_ignore_case(self, workspace: Path) -> None:
        # "MAIN" only appears in utils.py; "main" in main.py
        case_sensitive = grep_search("MAIN", workdir=workspace)
        case_insensitive = grep_search("MAIN", ignore_case=True, workdir=workspace)

        # Case-sensitive: should only match utils.py MAIN
        assert "utils.py" in str(case_sensitive)
        # Case-insensitive: should match both files
        assert "src/main.py" in str(case_insensitive)
        assert "utils.py" in str(case_insensitive)

    def test_include_filters_files(self, workspace: Path) -> None:
        result = grep_search("Project", include="*.md", workdir=workspace)
        text = str(result)
        assert "README.md" in text
        # Should NOT appear in .py files
        assert ".py" not in text.split("\n")[0]

    def test_include_py_only(self, workspace: Path) -> None:
        result = grep_search("hello", include="*.py", workdir=workspace)
        text = str(result)
        assert "src/main.py" in text
        assert "README.md" not in text

    def test_max_results_truncates(self, workspace: Path) -> None:
        # Create many matching lines
        big = workspace / "big.py"
        big.write_text("\n".join(f"match_{i}" for i in range(100)), encoding="utf-8")

        result = grep_search(r"match_\d+", max_results=5, workdir=workspace)
        text = str(result)
        # Should have exactly 5 match lines + summary
        lines = text.strip().split("\n")
        match_lines = [l for l in lines if l.startswith("big.py:")]
        assert len(match_lines) == 5
        assert "more matches" in text

    def test_excludes_venv_and_pycache(self, workspace: Path) -> None:
        result = grep_search("hello", workdir=workspace)
        text = str(result)
        assert ".venv" not in text
        assert "__pycache__" not in text

    def test_excludes_binary_files(self, workspace: Path) -> None:
        # Create a binary-ish file with matching content
        (workspace / "image.png").write_bytes(b"hello world PNG data")
        result = grep_search("hello", workdir=workspace)
        text = str(result)
        assert "image.png" not in text

    def test_workspace_boundary_escape(self, workspace: Path) -> None:
        with pytest.raises(ValueError, match="escapes workspace"):
            grep_search("hello", path="../", workdir=workspace)

    def test_search_subdirectory(self, workspace: Path) -> None:
        result = grep_search("helper", path="src", workdir=workspace)
        text = str(result)
        assert "utils.py" in text
        assert "README.md" not in text

    def test_no_matches(self, workspace: Path) -> None:
        result = grep_search("ZZZZNOTFOUND", workdir=workspace)
        text = str(result)
        assert "No matches" in text

    def test_path_not_found(self, workspace: Path) -> None:
        result = grep_search("hello", path="nonexistent_dir", workdir=workspace)
        text = str(result)
        assert "Error:" in text

    def test_output_format(self, workspace: Path) -> None:
        """Output should be file:linenum:content format."""
        result = grep_search("import os", workdir=workspace)
        text = str(result)
        lines = text.strip().split("\n")
        first_line = lines[0]
        # Should match pattern: path:linenum:content
        parts = first_line.split(":", 2)
        assert len(parts) >= 3
        assert parts[0] == "src/main.py"
        assert parts[1].isdigit()


# ---------------------------------------------------------------------------
# glob_search tests
# ---------------------------------------------------------------------------


class TestGlobSearch:
    def test_finds_matching_files(self, workspace: Path) -> None:
        result = glob_search("*.py", workdir=workspace)
        assert "src/main.py" in result
        assert "src/utils.py" in result
        assert "files found" in result

    def test_finds_md_files(self, workspace: Path) -> None:
        result = glob_search("*.md", workdir=workspace)
        assert "README.md" in result
        assert ".py" not in result.split("\n")[0]

    def test_recursive_pattern(self, workspace: Path) -> None:
        result = glob_search("**/*.py", workdir=workspace)
        assert "src/main.py" in result
        assert "src/utils.py" in result

    def test_specific_name(self, workspace: Path) -> None:
        result = glob_search("main.py", workdir=workspace)
        assert "src/main.py" in result

    def test_empty_pattern_returns_error(self, workspace: Path) -> None:
        result = glob_search("", workdir=workspace)
        assert result.startswith("Error:")

    def test_workspace_boundary_escape(self, workspace: Path) -> None:
        with pytest.raises(ValueError, match="escapes workspace"):
            glob_search("*.py", path="../../", workdir=workspace)

    def test_max_results_truncates(self, workspace: Path) -> None:
        # Create many files
        for i in range(20):
            (workspace / f"file_{i:03d}.txt").write_text("x", encoding="utf-8")

        result = glob_search("*.txt", max_results=5, workdir=workspace)
        lines = [l for l in result.strip().split("\n") if l.endswith(".txt")]
        assert len(lines) == 5
        assert "more files" in result

    def test_no_matches(self, workspace: Path) -> None:
        result = glob_search("*.xyz", workdir=workspace)
        assert "No files found" in result

    def test_excludes_venv(self, workspace: Path) -> None:
        result = glob_search("*.py", workdir=workspace)
        assert ".venv" not in result

    def test_path_not_found(self, workspace: Path) -> None:
        result = glob_search("*.py", path="nonexistent_dir", workdir=workspace)
        assert "Error:" in result

    def test_search_subdirectory(self, workspace: Path) -> None:
        result = glob_search("*.py", path="src", workdir=workspace)
        assert "main.py" in result
        assert "utils.py" in result
