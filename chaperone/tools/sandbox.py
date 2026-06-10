"""Sandboxed Python execution for on-the-fly bioinformatics (BioPython, parsing).

SECURITY BOUNDARY: this runs model-generated code. It is constrained to a workspace
directory, a wall-clock timeout, and a subprocess (no in-process eval), but it is
NOT a hardened sandbox — do not expose it to untrusted users without container/seccomp
isolation. Keep that limitation in mind before widening its capabilities.
"""

from __future__ import annotations

import os
import subprocess
import sys

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from chaperone.settings import get_settings
from chaperone.utils.logger import logger

_TIMEOUT_SECONDS = 120


class ScriptArgs(BaseModel):
    script_name: str = Field(description="Filename to write, e.g. 'parse.py'.")
    code: str = Field(description="Python source to execute.")


@tool(args_schema=ScriptArgs)
def execute_python_script(script_name: str, code: str) -> str:
    """Write and run a Python script in the sandbox workspace; return its stdout.

    Use for quick calculations, BioPython parsing, or data wrangling. Times out
    after 120s.
    """
    workspace = str(get_settings().paths.sandbox_dir)
    os.makedirs(workspace, exist_ok=True)
    path = os.path.join(workspace, script_name)
    with open(path, "w") as f:
        f.write(code)

    logger.info(f"Executing sandbox script: {path}")
    try:
        result = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            check=True,
            cwd=workspace,
        )
        return f"Execution successful.\nSTDOUT:\n{result.stdout}"
    except subprocess.CalledProcessError as e:
        logger.error(f"Sandbox script failed: {e.stderr}")
        return f"Execution failed.\nSTDERR:\n{e.stderr}"
    except subprocess.TimeoutExpired:
        logger.error("Sandbox script timed out.")
        return f"Execution timed out after {_TIMEOUT_SECONDS}s."
