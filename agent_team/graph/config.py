"""
Configuration constants and settings for the agent team system.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── LLM ─────────────────────────────────────
MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-4o")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))

# ── Guardrails ──────────────────────────────
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "10"))

# ── Workspace ───────────────────────────────
WORKSPACE_PATH: Path = Path(os.getenv("WORKSPACE_PATH", "./workspace")).resolve()
