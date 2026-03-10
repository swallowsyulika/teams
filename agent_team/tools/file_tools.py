"""
File system tools for expert agents.

Provides read_file and write_file as LangChain @tool functions.
All paths are resolved relative to the configured WORKSPACE_PATH.
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_core.tools import tool

from agent_team.graph.config import WORKSPACE_PATH


def _resolve_safe_path(path: str) -> Path:
    """Resolve *path* under WORKSPACE_PATH and reject traversal attempts."""
    resolved = (WORKSPACE_PATH / path).resolve()
    if not str(resolved).startswith(str(WORKSPACE_PATH)):
        raise ValueError(
            f"Path traversal detected: '{path}' resolves outside workspace."
        )
    return resolved


@tool
def read_file(path: str) -> str:
    """Read the content of a file at the given path (relative to workspace).

    Args:
        path: Relative file path inside the workspace directory.

    Returns:
        The file content as a UTF-8 string.
    """
    resolved = _resolve_safe_path(path)
    if not resolved.exists():
        return f"ERROR: File not found: {path}"
    if not resolved.is_file():
        return f"ERROR: Not a file: {path}"
    return resolved.read_text(encoding="utf-8")


@tool
def write_file(path: str, content: str) -> str:
    """Write (or overwrite) a file at the given path with the provided content.

    Parent directories are created automatically.

    Args:
        path: Relative file path inside the workspace directory.
        content: The content to write.

    Returns:
        A confirmation message.
    """
    resolved = _resolve_safe_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    return f"OK: wrote {len(content)} chars to {path}"
