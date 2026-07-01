"""The four built-in tools, plus the registry that exposes them to the model."""
from agent.tools import TOOL_SCHEMAS, TOOLS
from agent.tools.calculator import calculate
from agent.tools.files import read_file, write_file
from agent.tools.shell import run_shell


def test_registry_matches_schemas():
    assert sorted(TOOLS) == ["calculate", "read_file", "run_shell", "write_file"]
    assert [s["name"] for s in TOOL_SCHEMAS] == ["read_file", "write_file", "run_shell", "calculate"]


def test_calculate_basic_arithmetic():
    assert calculate("2 * (3 + 4) / 7") == "2.0"
    assert calculate("10 // 3") == "3"
    assert calculate("2 ** 8") == "256"


def test_calculate_returns_errors_instead_of_raising():
    assert "Error" in calculate("1/0")
    assert "Error" in calculate("import os")  # not a numeric expression


def test_file_write_then_read_roundtrip(tmp_path):
    target = tmp_path / "sub" / "note.txt"  # parent dir doesn't exist yet
    msg = write_file(str(target), "hello")
    assert "Wrote 5 characters" in msg
    assert read_file(str(target)) == "hello"


def test_read_missing_file_returns_error(tmp_path):
    assert "Error reading" in read_file(str(tmp_path / "nope.txt"))


def test_run_shell_captures_output():
    assert run_shell("echo hi").strip() == "hi"
