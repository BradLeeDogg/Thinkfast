from typing import Callable, Dict

from ..library import library_available
from .calculator import CALCULATE_SCHEMA, calculate
from .files import READ_FILE_SCHEMA, WRITE_FILE_SCHEMA, read_file, write_file
from .library import SEARCH_LIBRARY_SCHEMA, search_library
from .shell import RUN_SHELL_SCHEMA, run_shell

TOOL_SCHEMAS = [
    READ_FILE_SCHEMA,
    WRITE_FILE_SCHEMA,
    RUN_SHELL_SCHEMA,
    CALCULATE_SCHEMA,
]

TOOLS: Dict[str, Callable[..., str]] = {
    "read_file": read_file,
    "write_file": write_file,
    "run_shell": run_shell,
    "calculate": calculate,
}

# The library search tool only appears once you've indexed some PDFs (agent-index).
if library_available():
    TOOL_SCHEMAS.append(SEARCH_LIBRARY_SCHEMA)
    TOOLS["search_library"] = search_library
