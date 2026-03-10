"""
Bash tool for expert agents.

Runs shell commands in a subprocess restricted to the workspace directory.
"""

from __future__ import annotations

import subprocess

from langchain_core.tools import tool

from agent_team.graph.config import WORKSPACE_PATH


@tool
def bash(command: str) -> str:
    """Execute a shell command inside the workspace directory.

    The command runs with a 60-second timeout.  stdout and stderr are
    captured and returned together.

    Args:
        command: The shell command string to execute.

    Returns:
        Combined stdout + stderr output, or an error message on failure.
    """
    # Ensure the workspace directory exists
    WORKSPACE_PATH.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(WORKSPACE_PATH),
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[STDERR]\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n[EXIT CODE: {result.returncode}]"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: Command timed out after 60 seconds."
    except Exception as exc:
        return f"ERROR: {exc}"
